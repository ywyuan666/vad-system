"""
VAD 工具函数
=============

包含音频加载、格式转换、片段合并等通用功能。
"""

from typing import List, Tuple

import numpy as np


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """加载音频并重采样到目标采样率。

    Args:
        path: 音频文件路径（支持 wav / mp3 / flac / m4a 等）。
        target_sr: 目标采样率，默认 16kHz。

    Returns:
        (audio_array, sr) 其中 audio 为 (N,) 的 float32 数组，值域 [-1, 1]。
    """
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def frames_to_time(frame_idx: int, hop_length: int, sr: int) -> float:
    """将帧索引转换为时间（秒）。"""
    return frame_idx * hop_length / sr


def merge_segments(
    segments: List[Tuple[float, float]],
    min_duration: float = 0.1,
    max_silence: float = 0.5,
) -> List[Tuple[float, float]]:
    """合并邻近的语音段，去除过短片段，填充短暂静音。

    Args:
        segments: [(start, end), ...] 列表，单位秒。
        min_duration: 最短语音段长度，低于此值的段被丢弃。
        max_silence: 两段之间静音小于此值时合并为一段。

    Returns:
        合并后的 [(start, end), ...]。
    """
    if not segments:
        return []

    sorted_seg = sorted(segments, key=lambda x: x[0])
    merged: List[List[float]] = [list(sorted_seg[0])]

    for start, end in sorted_seg[1:]:
        if start - merged[-1][1] <= max_silence:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    merged_filtered = [(s, e) for s, e in merged if e - s >= min_duration]
    return merged_filtered


def segments_to_mask(
    segments: List[Tuple[float, float]],
    duration: float,
    hop_length: int,
    sr: int,
) -> np.ndarray:
    """将语音段列表转换为帧级别的 mask 数组。

    Returns:
        (T,) 的 bool 数组，True 表示该帧是语音。
    """
    n_frames = int(duration * sr / hop_length) + 1
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in segments:
        s_idx = int(start * sr / hop_length)
        e_idx = int(end * sr / hop_length)
        mask[s_idx:e_idx] = True
    return mask


def mask_to_segments(
    mask: np.ndarray,
    hop_length: int,
    sr: int,
) -> List[Tuple[float, float]]:
    """将帧级别的 bool mask 转换为 [(start, end), ...] 列表。"""
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = frames_to_time(i, hop_length, sr)
            while i < len(mask) and mask[i]:
                i += 1
            end = frames_to_time(i, hop_length, sr)
            segments.append((start, end))
        else:
            i += 1
    return segments


def save_segments_to_audio(
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    output_dir: str,
    prefix: str = "seg",
) -> List[str]:
    """将每个语音段保存为独立的 wav 文件。"""
    import os

    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for i, (start, end) in enumerate(segments):
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg_audio = audio[s_idx:e_idx]
        path = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
        sf.write(path, seg_audio, sr)
        paths.append(path)
    return paths
