"""
VAD 数据集模块
==============

提供 PyTorch Dataset，支持从以下来源构造训练数据：
    1. 标注的语音/非语音段列表
    2. Common Voice / DNS Challenge 等开源数据集
    3. 合成的含噪语音

每个样本是固定长度的 Fbank 片段 + 对应的帧级别标签。
"""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor
from .utils import segments_to_mask

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VADDataset(Dataset):
    """VAD 训练数据集。

    从音频 + 标注段构造固定长度的帧级别训练样本。
    """

    def __init__(
        self,
        audio_list: List[np.ndarray],
        label_segments_list: List[List[Tuple[float, float]]],
        sr: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 40,
        chunk_frames: int = 200,     # 每个样本的帧数（~2秒 @ 10ms）
        stride_frames: int = 100,    # 滑动步长
        augment: bool = True,
        noise_snr_range: Tuple[float, float] = (10, 30),
    ):
        """
        Args:
            audio_list: 音频数组列表。
            label_segments_list: 每个音频对应的标注段列表。
            sr: 采样率。
            hop_length: 帧移。
            win_length: 窗长。
            n_mels: Mel 滤波器组数量。
            chunk_frames: 每个样本包含的帧数。
            stride_frames: 滑动窗口步长。
            augment: 是否启用数据增强（加噪）。
            noise_snr_range: 加噪 SNR 范围 (min, max) dB。
        """
        if not HAS_TORCH:
            raise ImportError("VADDataset 需要 PyTorch: pip install torch")

        self.feat = FeatureExtractor(
            sr=sr, hop_length=hop_length, win_length=win_length, n_mels=n_mels,
        )
        self.chunk_frames = chunk_frames
        self.stride_frames = stride_frames
        self.sr = sr
        self.hop_length = hop_length
        self.augment = augment
        self.noise_snr_range = noise_snr_range

        # 预提取所有样本
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []  # (fbank, label)
        self._prepare_samples(audio_list, label_segments_list)

    def _prepare_samples(
        self,
        audio_list: List[np.ndarray],
        label_segments_list: List[List[Tuple[float, float]]],
    ) -> None:
        for audio, segments in zip(audio_list, label_segments_list):
            duration = len(audio) / self.sr
            fbank = self.feat.fbank(audio)  # (T, n_mels)
            label_mask = segments_to_mask(
                segments, duration, self.hop_length, self.sr
            )

            # 对齐
            min_len = min(fbank.shape[0], len(label_mask))
            fbank = fbank[:min_len]
            label_mask = label_mask[:min_len]

            # 滑动窗口切分
            for start in range(0, max(1, min_len - self.chunk_frames), self.stride_frames):
                end = start + self.chunk_frames
                if end > min_len:
                    # 末尾不足时 padding
                    fbank_chunk = np.zeros((self.chunk_frames, fbank.shape[1]))
                    fbank_chunk[:min_len - start] = fbank[start:]
                    label_chunk = np.zeros(self.chunk_frames, dtype=np.float32)
                    label_chunk[:min_len - start] = label_mask[start:].astype(float)
                else:
                    fbank_chunk = fbank[start:end]
                    label_chunk = label_mask[start:end].astype(float)

                self.samples.append((fbank_chunk, label_chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fbank, label = self.samples[idx]

        if self.augment:
            fbank = self._augment(fbank)

        # 归一化
        fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-10)

        return (
            torch.from_numpy(fbank).float(),
            torch.from_numpy(label).float(),
        )

    def _augment(self, fbank: np.ndarray) -> np.ndarray:
        """数据增强：Fbank 层面添加噪声。"""
        noise_std = random.uniform(0.005, 0.02)
        fbank = fbank + np.random.randn(*fbank.shape) * noise_std
        return fbank

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批处理打包函数。"""
        fbanks, labels = zip(*batch)
        fbank_shape = fbanks[0].shape
        labels_shape = labels[0].shape

        fbanks_pad = torch.stack([f.view(-1, fbank_shape[-1]) for f in fbanks])
        labels_pad = torch.stack([l.view(-1) for l in labels])
        return fbanks_pad, labels_pad
