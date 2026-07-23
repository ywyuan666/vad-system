"""
基于谱特征的 VAD
================

人声的频谱具有独特的结构特征：
  - 谱平坦度低（语音的频谱不平坦，共振峰结构明显）
  - 谱质心集中在 500-3000Hz（人声主要频段）
  - 谱能量集中在低频

通过组合多个谱特征进行判别，比纯能量法更鲁棒，
尤其在有背景音乐/环境噪声的场景下表现更好。
"""

from typing import List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor
from .utils import mask_to_segments, merge_segments, frames_to_time


class SpectralVAD:
    """基于谱特征的 VAD，结合谱平坦度 + 谱质心 + 能量。

    参考文献：
        - Ramirez, J., et al. "Voice activity detection. fundamentals and speech
          recognition system robustness." Robust Speech Recognition and Understanding (2007).
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        energy_weight: float = 0.4,
        flatness_weight: float = 0.4,
        centroid_weight: float = 0.2,
        # 各特征的阈值（自适应时自动计算）
        energy_thresh: Optional[float] = None,
        flatness_thresh: float = 0.6,      # 谱平坦度低于此值判定为语音
        centroid_low: float = 300,          # 谱质心下限 (Hz)
        centroid_high: float = 3500,        # 谱质心上限 (Hz)
        # 自适应参数
        adaptive: bool = True,
        noise_frames: int = 20,
        adaptive_ratio: float = 2.0,
        # 后处理
        smoothing_window: int = 5,
        min_speech_frames: int = 5,
        min_silence_frames: int = 15,
    ):
        self.feat = FeatureExtractor(sr=sr, hop_length=hop_length, win_length=win_length)
        self.sr = sr
        self.hop_length = hop_length
        self.energy_weight = energy_weight
        self.flatness_weight = flatness_weight
        self.centroid_weight = centroid_weight
        self.energy_thresh = energy_thresh
        self.flatness_thresh = flatness_thresh
        self.centroid_low = centroid_low
        self.centroid_high = centroid_high
        self.adaptive = adaptive
        self.noise_frames = noise_frames
        self.adaptive_ratio = adaptive_ratio
        self.smoothing_window = smoothing_window
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames

    def __call__(self, audio: np.ndarray, sr: Optional[int] = None) -> List[Tuple[float, float]]:
        if sr is not None and sr != self.feat.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.feat.sr)

        # 提取多种谱特征
        energy = self.feat.rms_energy(audio).squeeze(-1)       # (T,)
        flatness = self.feat.spectral_flatness(audio)          # (T,)
        centroid = self.feat.spectral_centroid(audio).squeeze(-1)  # (T,)

        speech_mask = self._detect(energy, flatness, centroid)
        speech_mask = self._post_process(speech_mask)

        segments = mask_to_segments(speech_mask, self.hop_length, self.feat.sr)
        return merge_segments(segments, min_duration=0.08, max_silence=0.3)

    def _detect(
        self,
        energy: np.ndarray,
        flatness: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """多特征融合决策。返回 0~1 的得分，>0.5 判为语音。"""
        T = min(len(energy), len(flatness), len(centroid))

        # 归一化特征到 [0, 1]
        e_norm = self._normalize(energy)
        f_norm = 1.0 - self._normalize(flatness)  # 平坦度越低越好
        c_score = ((centroid[:T] >= self.centroid_low) &
                   (centroid[:T] <= self.centroid_high)).astype(float)

        # 加权融合
        score = (
            self.energy_weight * e_norm[:T]
            + self.flatness_weight * f_norm[:T]
            + self.centroid_weight * c_score[:T]
        )
        return score > 0.5

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0,1]；自适应时使用噪声段估计。"""
        if self.adaptive:
            noise_x = x[: min(self.noise_frames, len(x))]
            x_min = noise_x.mean()
            x_max = x.max() + 1e-10
        else:
            x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def _post_process(self, mask: np.ndarray) -> np.ndarray:
        """中值平滑 + 去毛刺 + 填间隙。"""
        from scipy.ndimage import median_filter
        mask = median_filter(mask.astype(float), size=self.smoothing_window) > 0.5
        mask = self._remove_short(mask, self.min_speech_frames, True)
        mask = self._remove_short(mask, self.min_silence_frames, False)
        return mask

    @staticmethod
    def _remove_short(mask: np.ndarray, min_len: int, value: bool) -> np.ndarray:
        out = mask.copy()
        i = 0
        while i < len(out):
            if out[i] == value:
                j = i
                while j < len(out) and out[j] == value:
                    j += 1
                if j - i < min_len:
                    out[i:j] = not value
                i = j
            else:
                i += 1
        return out
