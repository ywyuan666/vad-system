"""
基于 DNN 的 VAD
===============

使用简单高效的 CNN + GRU 网络结构，输入 Fbank 特征，
输出帧级别的语音/非语音概率。

设计目标：
    - 轻量级：模型参数 < 100K，适合实时推理
    - 高性能：在干净/噪声环境下均优于传统方法
    - 可训练：用户可用自己的数据微调

网络结构：
    Input (T x 40 Fbank)
        ↓
    Conv1D (3x3, 32) + BN + ReLU
        ↓
    Conv1D (3x3, 64) + BN + ReLU
        ↓
    BiGRU (64) → Linear(64→1) → Sigmoid
        ↓
    Output (T x 1)  帧级别语音概率
"""

from typing import List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor
from .utils import (
    ensure_sr,
    mask_to_segments,
    merge_segments,
    remove_short,
)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── 模型定义 ──────────────────────────────────────────────────────────────


class VADNet(nn.Module):
    """轻量级 VAD 模型：Conv1D × 2 → BiGRU → Linear。

    Args:
        n_mels: Fbank 维度，默认 40。
        hidden: Conv 隐藏维度，默认 64。
        gru_hidden: GRU 隐藏维度，默认 64。
        dropout: Dropout 比例，默认 0.2。
    """

    def __init__(self, n_mels: int = 40, hidden: int = 64, gru_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden, gru_hidden, batch_first=True, bidirectional=True, num_layers=1,
        )
        self.classifier = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, T, n_mels) Fbank 特征。

        Returns:
            (B, T, 1) 帧级别语音概率。
        """
        x = x.transpose(1, 2)  # (B, n_mels, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, T, hidden)
        x, _ = self.gru(x)      # (B, T, gru_hidden*2)
        out = torch.sigmoid(self.classifier(x))  # (B, T, 1)
        return out


# ── DNNVAD 对外接口 ──────────────────────────────────────────────────────


class DNNVAD:
    """基于 DNN 的 VAD。

    使用 VADNet 模型进行帧级别的语音/非语音分类。

    Usage:
        >>> vad = DNNVAD(model_path="checkpoint.pt")
        >>> segments = vad(audio, sr=16000)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        sr: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 40,
        prob_threshold: float = 0.5,
        device: str = "cpu",
        min_speech_frames: int = 5,
        min_silence_frames: int = 10,
    ):
        if not HAS_TORCH:
            raise ImportError("DNNVAD 需要 PyTorch: pip install torch")

        self.feat = FeatureExtractor(
            sr=sr, hop_length=hop_length, win_length=win_length, n_mels=n_mels,
        )
        self.sr = sr
        self.hop_length = hop_length
        self.prob_threshold = prob_threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames

        self.device = torch.device(device)
        self.model = VADNet(n_mels=n_mels).to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load(model_path)

    def load(self, path: str) -> None:
        """加载预训练权重。"""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def save(self, path: str) -> None:
        """保存模型权重。"""
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def __call__(self, audio: np.ndarray, sr: Optional[int] = None) -> List[Tuple[float, float]]:
        """对输入音频进行 VAD 检测。

        Args:
            audio: 单声道音频，值域 [-1, 1]。
            sr: 采样率，省略则使用默认值。

        Returns:
            [(start, end), ...] 语音段列表。
        """
        # ── 边界保护 ────────────────────────────────────────────────
        if len(audio) == 0:
            return []
        if sr is not None and sr != self.feat.sr:
            audio = ensure_sr(audio, sr, self.feat.sr)

        # 提取 Fbank
        fbank = self.feat.fbank(audio)  # (T, n_mels)
        if fbank.shape[0] < 2:          # 至少需要 2 帧才能做有意义的 GRU 推理
            return []

        # 归一化
        fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-10)

        # 转为 Tensor
        x = torch.from_numpy(fbank).float().unsqueeze(0).to(self.device)  # (1, T, n_mels)

        # 推理
        probs = self.model(x).squeeze().cpu().numpy()  # (T,)

        # 阈值决策
        speech_mask = probs > self.prob_threshold

        # 后处理
        speech_mask = remove_short(speech_mask, self.min_speech_frames, True)
        speech_mask = remove_short(speech_mask, self.min_silence_frames, False)

        segments = mask_to_segments(speech_mask, self.hop_length, self.feat.sr)
        return merge_segments(segments, min_duration=0.08, max_silence=0.3)

    # ── 流式推理 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_frames(self, fbank: np.ndarray) -> np.ndarray:
        """对 Fbank 特征矩阵进行帧级别预测。

        Args:
            fbank: (T, n_mels) 特征矩阵。

        Returns:
            (T,) 语音概率数组。
        """
        if fbank.ndim == 1:
            fbank = fbank[np.newaxis, :]
        fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-10)
        x = torch.from_numpy(fbank).float().unsqueeze(0).to(self.device)
        probs = self.model(x).squeeze().cpu().numpy()
        return probs
