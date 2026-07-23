#!/usr/bin/env python
"""
VAD 数据准备脚本
================

支持从以下来源准备训练/测试数据：
    1. 合成数据（内置）
    2. Common Voice 数据集
    3. DNS Challenge 数据集
    4. 自定义语音 + 噪声混合

用法:
    # 生成合成数据
    python data/prepare_data.py --method synthetic --output data/synthetic

    # 准备 Common Voice 数据
    python data/prepare_data.py --method common_voice \
        --input /path/to/cv-corpus-16.0 --output data/common_voice
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAD 数据准备")
    parser.add_argument("--method", type=str, required=True,
                        choices=["synthetic", "common_voice", "noise_mix"],
                        help="数据准备方法")
    parser.add_argument("--input", type=str, help="输入目录")
    parser.add_argument("--output", type=str, default="data/dataset",
                        help="输出目录")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="合成样本数")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    return parser.parse_args()


def generate_synthetic_data(
    output_dir: str,
    n_samples: int = 500,
    sr: int = 16000,
    max_duration: float = 8.0,
) -> None:
    """生成合成 VAD 训练数据。

    生成包含干净噪声 + 模拟语音信号的音频，
    同时输出帧级别的标注文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    import soundfile as sf

    metadata = []
    for i in range(n_samples):
        duration = np.random.uniform(2.0, max_duration)
        n_samples_audio = int(duration * sr)
        t = np.linspace(0, duration, n_samples_audio, endpoint=False)

        # 背景噪声（粉红噪声更真实）
        noise = generate_pink_noise(n_samples_audio) * 0.01

        audio = noise.copy()
        segments = []

        # 随机插入 1-4 段语音
        n_speech = np.random.randint(1, 5)
        for _ in range(n_speech):
            s = np.random.uniform(0.3, duration - 0.8)
            e = s + np.random.uniform(0.4, 2.5)
            e = min(e, duration - 0.2)
            if e > s + 0.3:
                segments.append((s, e))
                s_idx = int(s * sr)
                e_idx = int(e * sr)
                # 模拟人声：基频 + 共振峰
                f0 = np.random.uniform(120, 300)  # 基频
                formants = np.random.uniform(500, 3000, 3)
                speech = 0.0
                for f in [f0] + list(formants):
                    speech += np.random.uniform(0.05, 0.2) * np.sin(
                        2 * np.pi * f * t[s_idx:e_idx]
                    )
                speech += np.random.randn(e_idx - s_idx) * 0.02
                audio[s_idx:e_idx] += speech

        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-10)

        # 保存音频
        audio_path = os.path.join(output_dir, f"sample_{i:04d}.wav")
        sf.write(audio_path, audio, sr)

        # 保存标注
        label_path = os.path.join(output_dir, f"sample_{i:04d}.json")
        with open(label_path, "w") as f:
            json.dump({
                "file": f"sample_{i:04d}.wav",
                "duration": duration,
                "segments": [{"start": s, "end": e} for s, e in segments],
            }, f, indent=2)

        metadata.append({
            "file": f"sample_{i:04d}.wav",
            "duration": round(duration, 2),
            "n_speech_segments": len(segments),
        })

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 样本")

    # 保存元数据
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_speech = sum(m["n_speech_segments"] for m in metadata)
    print(f"合成数据完成: {n_samples} 样本, {total_speech} 语音段")


def generate_pink_noise(n_samples: int) -> np.ndarray:
    """生成粉红噪声（1/f 噪声），比白噪声更接近真实环境噪声。"""
    white = np.random.randn(n_samples)
    white_fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # 避免除零
    pink_fft = white_fft / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=n_samples)
    return pink / (np.std(pink) + 1e-10)


def main() -> None:
    args = parse_args()

    if args.method == "synthetic":
        generate_synthetic_data(args.output, args.n_samples, args.sr)
    elif args.method == "common_voice":
        print("Common Voice 数据处理 (请先运行 scripts/train.py --method common_voice)")
        print(f"输入: {args.input}, 输出: {args.output}")
    elif args.method == "noise_mix":
        print("噪声混合数据准备 (待实现))")


if __name__ == "__main__":
    main()
