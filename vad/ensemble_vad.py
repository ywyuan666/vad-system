"""
集成 VAD + 在线噪声自适应阈值
===============================
通过投票/加权融合多种 VAD 方法，结合实时噪声估计实现自适应阈值。

"单一模型总有不擅长的场景，集成是工业界的标准做法。"

策略:
  1. VotingEnsemble: 多数投票 (≥2/3 方法判定为语音则判定为语音)
  2. WeightedEnsemble: 加权融合 (每个方法根据其历史置信度加权)
  3. AdaptiveThreshold: 基于实时背景噪声估计动态调整阈值

用法:
  from vad.ensemble_vad import VotingEnsemble, AdaptiveThreshold

  # 多数投票集成
  ensemble = VotingEnsemble(min_votes=2)
  segments = ensemble.detect(audio)

  # 自适应阈值
  adaptive = AdaptiveThreshold(base_vad=dnn_vad)
  segments = adaptive.detect(audio)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from vad import EnergyVAD, SpectralVAD, DNNVAD
from vad.utils import merge_segments, remove_short


# ── 在线噪声估计 ──────────────────────────────────────────────────────────


class OnlineNoiseEstimator:
    """
    在线噪声估计器。

    通过跟踪音频中的"静音段"（能量最低的 N% 帧）来实时估计背景噪声水平。
    支持滑动窗口更新，适用于实时流式场景。

    原理: 假设音频中大部分时间是静音/噪声，取最低能量的帧作为噪声估计。
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_ms: int = 20,
        noise_percentile: float = 15.0,
        smoothing: float = 0.9,
    ):
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.noise_percentile = noise_percentile
        self.smoothing = smoothing
        self.noise_energy: Optional[float] = None
        self.energy_buffer: list[float] = field(default_factory=list)
        self.max_buffer = 1000

    def update(self, frame: np.ndarray) -> float:
        """更新噪声估计并返回当前帧的 SNR 估计 (dB)。"""
        # 计算帧能量 (RMS)
        energy = float(np.sqrt(np.mean(frame ** 2)) + 1e-10)
        self.energy_buffer.append(energy)
        if len(self.energy_buffer) > self.max_buffer:
            self.energy_buffer.pop(0)

        # 估计噪声能量: 取最近帧的最低百分位
        if len(self.energy_buffer) > 10:
            current_noise = float(np.percentile(self.energy_buffer, self.noise_percentile))
        else:
            current_noise = energy

        # 平滑
        if self.noise_energy is None:
            self.noise_energy = current_noise
        else:
            self.noise_energy = (
                self.smoothing * self.noise_energy + (1 - self.smoothing) * current_noise
            )

        # 估计 SNR
        snr_db = 20 * np.log10(energy / (self.noise_energy + 1e-10))
        return float(snr_db)

    def get_noise_floor(self) -> float:
        """获取当前估计的噪声底噪。"""
        return self.noise_energy or 0.01

    def reset(self) -> None:
        self.noise_energy = None
        self.energy_buffer.clear()


# ── 自适应阈值 VAD ────────────────────────────────────────────────────────


class AdaptiveThresholdVAD:
    """
    基于实时噪声估计的动态阈值 VAD。

    核心思想: 噪声大时抬高阈值（降低虚警），噪声小时降低阈值（提高召回）。
    阈值调整策略: threshold = base_threshold + noise_bias
    其中 noise_bias 基于当前 SNR 动态计算。

    这比固定阈值更鲁棒——在安静环境下不会漏检，在嘈杂环境下不会虚警。
    """

    def __init__(
        self,
        base_vad: Optional[DNNVAD] = None,
        base_threshold: float = 0.5,
        sr: int = 16000,
        frame_ms: int = 20,
        min_threshold: float = 0.3,
        max_threshold: float = 0.75,
    ):
        self.base_vad = base_vad or DNNVAD()
        self.base_threshold = base_threshold
        self.sr = sr
        self.frame_ms = frame_ms
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.noise_estimator = OnlineNoiseEstimator(sr=sr, frame_ms=frame_ms)

    def _compute_adaptive_threshold(self, snr_db: float) -> float:
        """
        根据当前 SNR 计算自适应阈值。

        规则 (经验设计):
          - SNR > 20dB (安静): threshold = base (0.5)
          - SNR 5~20dB (中等噪声): threshold = base + 0.1~0.15
          - SNR < 5dB (高噪声): threshold = max (0.75)
        """
        if snr_db > 20:
            bias = 0.0
        elif snr_db > 10:
            bias = 0.05 + (20 - snr_db) * 0.005
        elif snr_db > 5:
            bias = 0.1 + (10 - snr_db) * 0.01
        else:
            bias = 0.25

        threshold = self.base_threshold + bias
        return float(np.clip(threshold, self.min_threshold, self.max_threshold))

    def detect(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """
        使用自适应阈值进行 VAD 检测。

        流程:
          1. 将音频分帧
          2. 对每一帧，用 DNN VAD 计算语音概率
          3. 根据噪声估计调整阈值
          4. 用自适应阈值判决
          5. 后处理 (去毛刺 + 合并)
        """
        import torch
        from vad.feature_extractor import FeatureExtractor

        self.noise_estimator.reset()
        feat_ext = FeatureExtractor()

        # 提取 Fbank (帧级特征)
        fbank = feat_ext.extract_fbank(audio)  # (T, n_mels)

        # 分帧计算能量用于噪声估计
        frame_len = int(self.sr * self.frame_ms / 1000)
        n_frames = len(audio) // frame_len
        frame_energies = []
        for i in range(n_frames):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            frame_energies.append(float(np.sqrt(np.mean(frame ** 2)) + 1e-10))

        # DNN 推理
        model = self.base_vad.model
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(fbank).unsqueeze(0)
            probs = model(x)[0, :, 0].numpy()  # (T,)

        # Fbank 帧数通常与音频帧数不同，做对齐
        fbank_frames = len(probs)
        aligned_energies = np.interp(
            np.linspace(0, len(frame_energies) - 1, fbank_frames),
            np.arange(len(frame_energies)),
            frame_energies,
        )

        # 自适应阈值判決
        speech_mask = np.zeros(fbank_frames, dtype=bool)
        for t in range(fbank_frames):
            # 更新噪声估计
            snr = self.noise_estimator.update(
                np.array([aligned_energies[t]], dtype=np.float32)
            )
            threshold = self._compute_adaptive_threshold(snr)
            speech_mask[t] = probs[t] > threshold

        # 后处理: 中值滤波 + 去毛刺 + 合并
        from scipy.ndimage import median_filter
        speech_mask = median_filter(speech_mask.astype(float), size=3) > 0.5

        # 转 segments
        from vad.utils import mask_to_segments
        segments = mask_to_segments(speech_mask, self.sr)
        segments = remove_short(segments, min_dur=0.05)
        segments = merge_segments(segments, max_silence=0.5)

        return segments


# ── 集成 VAD ──────────────────────────────────────────────────────────────


@dataclass
class EnsembleVAD:
    """
    多 VAD 方法集成检测器。

    策略:
      - 'voting': 多数投票 (threshold 控制需要多少方法同意)
      - 'weighted': 加权融合 (每个方法的历史 F1 作为权重)
      - 'or': 任一方法判定为语音 → 语音 (高召回)
      - 'and': 所有方法判定为语音 → 语音 (高精度)

    使用场景:
      - voting: 通用场景，平衡精度/召回
      - or: 安全场景（如语音唤醒），宁可虚警不可漏检
      - and: 高精度场景（如会议转录），宁可漏检不可虚警
    """

    methods: list = field(default_factory=lambda: [
        ("energy", EnergyVAD()),
        ("spectral", SpectralVAD()),
        ("dnn", DNNVAD()),
    ])
    strategy: str = "voting"
    voting_threshold: float = 0.5  # 多数投票时需要的比例 (0.5 = 2/3)
    weights: Optional[list[float]] = None  # weighted 模式下各方法的权重

    def detect(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """集成检测。"""
        from vad.utils import mask_to_segments

        # 收集各方法的帧级判决
        masks = []
        weights = self.weights or [1.0 / len(self.methods)] * len(self.methods)

        for name, vad in self.methods:
            segments = vad.detect(audio)
            mask = self._segments_to_mask(segments, len(audio))
            masks.append(mask)

        # 融合
        n_methods = len(masks)
        if self.strategy == "voting":
            # 多数投票
            n_agreed = np.sum(masks, axis=0)
            fused = n_agreed >= self.voting_threshold * n_methods

        elif self.strategy == "weighted":
            # 加权融合
            weighted_sum = np.zeros(len(audio), dtype=float)
            for mask, w in zip(masks, weights):
                weighted_sum += mask.astype(float) * w
            fused = weighted_sum >= 0.5

        elif self.strategy == "or":
            fused = np.any(masks, axis=0)

        elif self.strategy == "and":
            fused = np.all(masks, axis=0)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # 后处理
        from scipy.ndimage import median_filter
        fused = median_filter(fused.astype(float), size=5) > 0.5

        segments = mask_to_segments(fused, sr=16000)
        segments = remove_short(segments, min_dur=0.05)
        segments = merge_segments(segments, max_silence=0.5)

        return segments

    @staticmethod
    def _segments_to_mask(
        segments: list[tuple[float, float]], n_samples: int, sr: int = 16000
    ) -> np.ndarray:
        """段列表 → 采样点级 mask。"""
        mask = np.zeros(n_samples, dtype=bool)
        for s, e in segments:
            si = int(s * sr)
            ei = min(int(e * sr), n_samples)
            if si < ei:
                mask[si:ei] = True
        return mask
