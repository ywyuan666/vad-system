"""
通用声学特征提取模块
====================

提供 librosa / torchaudio 后端的 Fbank / MFCC / 谱特征提取，
供各 VAD 模块共用。
"""

from typing import Optional

import numpy as np

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    import torchaudio

    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class FeatureExtractor:
    """声学特征提取器，支持 librosa 和 torchaudio 两种后端。"""

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 40,
        backend: str = "librosa",
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.backend = backend

    def fbank(self, audio: np.ndarray) -> np.ndarray:
        """提取 Fbank 特征，shape: (T, n_mels)。"""
        if self.backend == "librosa" or not HAS_TORCHAUDIO:
            return self._fbank_librosa(audio)
        return self._fbank_torchaudio(audio)

    def _fbank_librosa(self, audio: np.ndarray) -> np.ndarray:
        if not HAS_LIBROSA:
            raise ImportError("请安装 librosa: pip install librosa")
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr // 2,
        )
        log_S = librosa.power_to_db(S, ref=np.max, top_db=None).T  # (T, n_mels)
        return log_S

    def _fbank_torchaudio(self, audio: np.ndarray) -> np.ndarray:
        if not HAS_TORCHAUDIO:
            raise ImportError("请安装 torchaudio: pip install torchaudio")
        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=self.n_mels,
            sample_frequency=self.sr,
            frame_length=self.win_length / self.sr * 1000,
            frame_shift=self.hop_length / self.sr * 1000,
            window_type="hamming",
            use_energy=False,
        )
        return fbank.numpy()  # (T, n_mels)

    def spectral_flatness(self, audio: np.ndarray) -> np.ndarray:
        """谱平坦度，shape: (T,)。值越接近 1 越接近噪声。"""
        if not HAS_LIBROSA:
            raise ImportError("请安装 librosa: pip install librosa")
        S = np.abs(
            librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )
        )
        geometric = np.exp(np.mean(np.log(S + 1e-10), axis=0))
        arithmetic = np.mean(S, axis=0) + 1e-10
        return (geometric / arithmetic).T  # (T,)

    def spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """谱质心，shape: (T,)。语音通常集中在 500-3000 Hz。"""
        if not HAS_LIBROSA:
            raise ImportError("请安装 librosa: pip install librosa")
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        return centroid.T  # (T, 1)

    def rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """短时 RMS 能量，shape: (T, 1)。"""
        if not HAS_LIBROSA:
            raise ImportError("请安装 librosa: pip install librosa")
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )
        return rms.T  # (T, 1)

    def zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """短时过零率，shape: (T, 1)。语音的 ZCR 通常高于静音。"""
        if not HAS_LIBROSA:
            raise ImportError("请安装 librosa: pip install librosa")
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )
        return zcr.T  # (T, 1)
