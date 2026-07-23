#!/usr/bin/env python
"""
VAD 方法对比评测
================

在同一组测试数据上对比 Energy / Spectral / DNN 三种 VAD 方法的性能。

用法:
    # 使用合成数据评测
    python benchmark/benchmark.py --method synthetic

    # 使用自定义标注数据评测
    python benchmark/benchmark.py --data_dir data/test --labels labels.json
"""

import argparse
import json
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vad import EnergyVAD, SpectralVAD, DNNVAD
from vad.evaluator import VADEvaluator
from vad.utils import load_audio

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAD 方法对比评测")
    parser.add_argument("--method", type=str, default="synthetic",
                        choices=["synthetic", "custom"],
                        help="评测数据来源")
    parser.add_argument("--data_dir", type=str, help="数据目录")
    parser.add_argument("--labels", type=str, help="标注文件")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="合成测试样本数")
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--dnn_model", type=str, help="DNN 模型路径")
    return parser.parse_args()


def create_synthetic_test_data(n_samples: int, sr: int = 16000) -> list:  # type: ignore[type-arg]
    """生成合成测试数据。"""
    from vad.utils import mask_to_segments

    test_cases = []
    for i in range(n_samples):
        duration = np.random.uniform(2.0, 6.0)
        n = int(duration * sr)
        t = np.linspace(0, duration, n, endpoint=False)

        # 背景噪声
        noise = np.random.randn(n) * 0.008
        audio = noise.copy()
        segments = []

        n_speech = np.random.randint(1, 4)
        for _ in range(n_speech):
            s = np.random.uniform(0.3, duration - 0.8)
            e = s + np.random.uniform(0.4, 2.0)
            e = min(e, duration - 0.2)
            if e > s + 0.3:
                segments.append((s, e))
                s_idx = int(s * sr)
                e_idx = int(e * sr)
                speech = (
                    0.2 * np.sin(2 * np.pi * 200 * t[s_idx:e_idx])
                    + 0.1 * np.sin(2 * np.pi * 1200 * t[s_idx:e_idx])
                    + 0.05 * np.random.randn(e_idx - s_idx)
                )
                audio[s_idx:e_idx] += speech

        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        test_cases.append({
            "audio": audio,
            "segments": segments,
            "sr": sr,
            "name": f"sample_{i:04d}",
        })

    return test_cases


def create_vad_instances(dnn_model: str = None):
    """创建所有 VAD 实例。"""
    vads = {
        "Energy (adaptive)": EnergyVAD(mode="adaptive", adaptive_ratio=2.5),
        "Energy (fixed)": EnergyVAD(mode="fixed", energy_thresh=0.02),
        "Spectral": SpectralVAD(),
    }
    if dnn_model and os.path.exists(dnn_model):
        vads["DNN"] = DNNVAD(model_path=dnn_model)
    return vads


def run_benchmark(
    test_cases: list,  # type: ignore[type-arg]
    vads: dict,  # type: ignore[type-arg]
    output_dir: str,
) -> pd.DataFrame:
    """运行对比评测。"""
    os.makedirs(output_dir, exist_ok=True)
    evaluator = VADEvaluator()

    results = []
    for vad_name, vad_fn in vads.items():
        print(f"\n{'='*50}")
        print(f"评测: {vad_name}")
        print("=" * 50)

        frame_f1s = []
        seg_detections = []
        latencies = []
        audios_duration = []

        for case in test_cases:
            audio = case["audio"]
            label_segments = case["segments"]

            # 测延迟和 RTF（实时率 = 处理耗时 / 音频时长）
            audio_duration = len(audio) / case["sr"]
            t0 = time.perf_counter()
            hyp_segments = vad_fn(audio, case["sr"])
            proc_time = time.perf_counter() - t0
            latency = proc_time * 1000  # ms
            rtf = proc_time / audio_duration if audio_duration > 0 else 0

            try:
                result = evaluator.evaluate(
                    lambda a, sr: vad_fn(a, sr), audio, label_segments
                )
                frame_f1s.append(result.frame.f1_score)
                seg_detections.append(result.segment.detection_rate)
                latencies.append(latency)
                audios_duration.append(audio_duration)
            except Exception as e:
                print(f"  评测异常: {e}")

        # 汇总
        avg_f1 = np.mean(frame_f1s) if frame_f1s else 0
        avg_det = np.mean(seg_detections) if seg_detections else 0
        avg_lat = np.mean(latencies) if latencies else 0
        avg_rtf = np.mean(latencies) / 1000 / np.mean(audios_duration) if audios_duration else 0

        print(f"  帧级别 F1:     {avg_f1:.4f}")
        print(f"  段检测率:      {avg_det:.4f}")
        print(f"  平均延迟:      {avg_lat:.2f} ms  |  RTF: {avg_rtf:.4f}")

        results.append({
            "VAD 方法": vad_name,
            "帧级别 F1": round(avg_f1, 4),
            "段检测率": round(avg_det, 4),
            "延迟 (ms)": round(avg_lat, 2),
            "RTF": round(avg_rtf, 4),
        })

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果保存: {csv_path}")
    print(df.to_string(index=False))

    # 绘制柱状图
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        metrics = ["帧级别 F1", "段检测率", "RTF"]
        for ax, metric in zip(axes, metrics):
            vals = [r[metric] for r in results]
            names = [r["VAD 方法"] for r in results]
            bars = ax.bar(names, vals, color=["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"])
            ax.set_title(metric)
            ax.set_ylim(0, max(vals) * 1.2 if metric != "RTF" else max(vals) * 1.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "benchmark_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"对比图保存: {plot_path}")

    return df


def main() -> None:
    args = parse_args()

    if args.method == "synthetic":
        print(f"生成 {args.n_samples} 个合成测试样本...")
        test_cases = create_synthetic_test_data(args.n_samples)
    elif args.method == "custom":
        if not args.data_dir:
            print("错误: custom 模式需要 --data_dir")
            return
        # 加载自定义测试数据
        test_cases = []
        for ext in ("*.wav", "*.mp3"):
            for path in Path(args.data_dir).rglob(ext):
                audio, sr = load_audio(str(path))
                # 如果没有标注，使用整个音频作为语音
                test_cases.append({
                    "audio": audio,
                    "segments": [(0.0, len(audio) / sr)],
                    "sr": sr,
                    "name": path.stem,
                })
        print(f"加载 {len(test_cases)} 个测试文件")

    vads = create_vad_instances(args.dnn_model)
    run_benchmark(test_cases, vads, args.output)


if __name__ == "__main__":
    main()
