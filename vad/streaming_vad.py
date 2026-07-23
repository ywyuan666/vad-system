"""
流式 VAD (Streaming VAD)
========================

支持流式音频输入的 VAD，具备：
    - 基于滑动窗口的实时检测
    - 状态机管理（静音→语音→静音）
    - 可配置的 hangover（说话结束后维持语音状态 N 帧，避免切句）
    - 低延迟（取决于窗口大小和 hop）

适用于实时通信、语音唤醒、流式 ASR 等场景。
"""

import numpy as np
from typing import Callable, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field


class VADState(Enum):
    SILENCE = 0
    SPEECH = 1
    HANGOVER = 2  # 语音结束后的保持期


@dataclass
class StreamingVADConfig:
    """流式 VAD 配置。"""
    frame_ms: int = 20           # 每帧时长 (ms)
    window_ms: int = 200         # 滑动窗口长度 (ms)
    hop_ms: int = 100            # 滑动步长 (ms)
    speech_threshold: float = 0.5
    hangover_frames: int = 10    # 语音结束后保持帧数
    min_speech_frames: int = 5
    min_silence_frames: int = 10


class StreamingVAD:
    """流式 VAD 封装。

    支持任意底层 VAD 方法（energy / spectral / dnn），
    通过 state_detector 回调进行帧级检测。
    """

    def __init__(
        self,
        state_detector: Callable[[np.ndarray, int], np.ndarray],
        config: StreamingVADConfig = None,
        sr: int = 16000,
    ):
        """
        Args:
            state_detector: 接收 (audio_chunk, sr) 返回帧级别 bool mask 的函数。
                可以传入 EnergyVAD / SpectralVAD / DNNVAD 实例的 predict_frames 或自定义函数。
            config: 流式配置。
            sr: 采样率。
        """
        self.detector = state_detector
        self.config = config or StreamingVADConfig()
        self.sr = sr
        self.frame_len = int(self.config.frame_ms * sr / 1000)
        self.window_len = int(self.config.window_ms * sr / 1000)
        self.hop_len = int(self.config.hop_ms * sr / 1000)

        # 流式状态
        self._buffer = np.array([], dtype=np.float32)
        self._state = VADState.SILENCE
        self._hangover_counter = 0
        self._segments: List[Tuple[float, float]] = []
        self._current_start: float = 0.0
        self._total_processed_ms: float = 0.0

    def reset(self):
        """重置流式状态（开始新音频流时调用）。"""
        self._buffer = np.array([], dtype=np.float32)
        self._state = VADState.SILENCE
        self._hangover_counter = 0
        self._segments = []
        self._current_start = 0.0
        self._total_processed_ms = 0.0

    @property
    def segments(self) -> List[Tuple[float, float]]:
        """返回当前已检测到的所有语音段。"""
        return self._segments

    def process_chunk(self, chunk: np.ndarray) -> Tuple[VADState, Optional[Tuple[float, float]]]:
        """处理一帧音频数据。

        Args:
            chunk: 当前帧音频数据。

        Returns:
            (当前状态, 可能的语音段边界)。
            当检测到完整语音段时返回 (state, (start, end))，否则 state 保持不变。
        """
        self._buffer = np.concatenate([self._buffer, chunk])

        # 累积足够数据后才检测
        if len(self._buffer) < self.window_len:
            return self._state, None

        # 取滑动窗口
        window = self._buffer[:self.window_len]
        self._buffer = self._buffer[self.hop_len:]

        # 帧级检测
        mask = self.detector(window, self.sr)
        # 取窗口内多数决策
        is_speech = mask.mean() > self.config.speech_threshold

        offset_ms = self._total_processed_ms

        result = None

        if self._state == VADState.SILENCE:
            if is_speech:
                self._state = VADState.SPEECH
                self._current_start = offset_ms / 1000.0

        elif self._state == VADState.SPEECH:
            if not is_speech:
                self._state = VADState.HANGOVER
                self._hangover_counter = 0

        elif self._state == VADState.HANGOVER:
            if is_speech:
                self._state = VADState.SPEECH
                self._hangover_counter = 0
            else:
                self._hangover_counter += 1
                if self._hangover_counter >= self.config.hangover_frames:
                    # 语音段结束
                    end = offset_ms / 1000.0
                    if end - self._current_start >= 0.1:
                        self._segments.append((self._current_start, end))
                        result = (VADState.SILENCE, (self._current_start, end))
                    self._state = VADState.SILENCE

        self._total_processed_ms += self.config.hop_ms
        return self._state, result

    def finish(self) -> List[Tuple[float, float]]:
        """结束流式处理，返回最终语音段列表。"""
        if self._state in (VADState.SPEECH, VADState.HANGOVER):
            end = self._total_processed_ms / 1000.0
            if end - self._current_start >= 0.1:
                self._segments.append((self._current_start, end))
        return self._segments
