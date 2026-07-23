"""
VAD — Voice Activity Detection System
======================================

支持三种 VAD 方法：
  1. EnergyVAD       — 基于短时能量的阈值检测
  2. SpectralVAD     — 基于谱特征的鲁棒检测
  3. DNNVAD          — 基于 DNN 的检测（训练/推理）

所有 VAD 统一暴露 `__call__(audio, sr) -> segments` 接口。
"""

from .energy_vad import EnergyVAD
from .spectral_vad import SpectralVAD
from .dnn_vad import DNNVAD
from .streaming_vad import StreamingVAD
from .evaluator import VADEvaluator
from .feature_extractor import FeatureExtractor
from .ensemble_vad import EnsembleVAD, AdaptiveThresholdVAD

__all__ = [
    "EnergyVAD",
    "SpectralVAD",
    "DNNVAD",
    "StreamingVAD",
    "VADEvaluator",
    "FeatureExtractor",
    "EnsembleVAD",
    "AdaptiveThresholdVAD",
]
