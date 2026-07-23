"""
基于短时能量的 VAD
==================

核心思想：人声的能量(振幅)通常显著高于静音/背景噪声。
通过自适应或固定阈值进行判别，计算量极小，适合实时/嵌入式场景。

优点：速度快、实现简单、资源消耗低。
缺点：对非平稳噪声敏感、阈值需要针对场景调整。
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


class EnergyVAD:
    """基于短时能量 + 过零率的 VAD。

    支持两种阈值模式：
        - fixed:     手动指定 energy_thresh 和 zcr_thresh
        - adaptive:  根据前 N 帧静音段自动计算阈值（推荐）
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        mode: str = "adaptive",
        energy_thresh: Optional[float] = None,
        zcr_thresh: Optional[float] = None,
        zcr_enabled: bool = False,
        smoothing_window: int = 3,       # 中值滤波窗口大小
        min_speech_frames: int = 5,      # 最小语音帧数（过滤毛刺）
        min_silence_frames: int = 10,    # 最大静音帧数（填充短间隙）
        adaptive_ratio: float = 2.5,     # 自适应阈值 = 背景噪声能量 * 该系数
        noise_frames: int = 20,          # 用于估计噪声的前 N 帧
    ):
        self.feat = FeatureExtractor(sr=sr, hop_length=hop_length, win_length=win_length)
        self.sr = sr
        self.hop_length = hop_length
        self.mode = mode
        self.energy_thresh = energy_thresh
        self.zcr_thresh = zcr_thresh
        self.zcr_enabled = zcr_enabled
        self.smoothing_window = smoothing_window
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.adaptive_ratio = adaptive_ratio
        self.noise_frames = noise_frames

    def __call__(self, audio: np.ndarray, sr: Optional[int] = None) -> List[Tuple[float, float]]:
        """对输入音频进行 VAD 检测。

        Args:
            audio: 单声道音频数组，值域 [-1, 1]。
            sr: 采样率，省略则使用初始化时的 sr。

        Returns:
            [(start_sec, end_sec), ...] 语音段列表。
        """
        # ── 边界保护 ────────────────────────────────────────────────
        if len(audio) == 0:
            return []
        if sr is not None and sr != self.feat.sr:
            audio = ensure_sr(audio, sr, self.feat.sr)

        # 提取能量和过零率
        energy = self.feat.rms_energy(audio).squeeze(-1)  # (T,)
        if len(energy) < 3:
            return []

        speech_mask = self._detect(audio, energy)

        # 后处理
        speech_mask = self._smooth(speech_mask)
        speech_mask = self._remove_spikes(speech_mask)
        speech_mask = self._fill_gaps(speech_mask)

        segments = mask_to_segments(speech_mask, self.hop_length, self.feat.sr)
        return merge_segments(segments, min_duration=0.08, max_silence=0.3)

    def _detect(self, audio: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """核心检测逻辑。"""
        if self.mode == "fixed" and self.energy_thresh is not None:
            thresh = self.energy_thresh
        else:
            # 自适应阈值：取前 noise_frames 帧能量均值 * adaptive_ratio
            noise_energy = np.mean(energy[: min(self.noise_frames, len(energy))]) + 1e-10
            thresh = noise_energy * self.adaptive_ratio

        speech = energy > thresh

        # 可选：联合过零率进一步过滤
        if self.zcr_enabled:
            zcr = self.feat.zero_crossing_rate(audio).squeeze(-1)
            # 语音的过零率通常高于静音（因为语音包含丰富的清/浊音交替）
            zcr_thresh = self.zcr_thresh or np.median(zcr[:self.noise_frames]) * 2
            speech = speech & (zcr > zcr_thresh)

        return speech

    def _smooth(self, mask: np.ndarray) -> np.ndarray:
        """中值滤波平滑。"""
        from scipy.ndimage import median_filter
        return median_filter(mask.astype(float), size=self.smoothing_window) > 0.5

    def _remove_spikes(self, mask: np.ndarray) -> np.ndarray:
        """移除过短的语音毛刺。"""
        return remove_short(mask, self.min_speech_frames, True)

    def _fill_gaps(self, mask: np.ndarray) -> np.ndarray:
        """填充过短的静音间隙。"""
        return remove_short(~mask, self.min_silence_frames, False)
