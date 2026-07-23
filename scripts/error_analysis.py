#!/usr/bin/env python3
"""
VAD 错误 / 失败模式分析工具
==============================
分析 VAD 检测结果中的错误模式（False Alarm / Miss），
帮助理解模型在哪些场景下容易失败，为模型优化提供方向。

用法:
  python scripts/error_analysis.py
  python scripts/error_analysis.py --audio_dir ./test_audio --label_dir ./test_labels
  python scripts/error_analysis.py --generate_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── 项目导入 ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD, VADEvaluator
from vad.utils import load_audio, ensure_sr, segments_to_mask


def generate_synthetic_test_set(
    num_samples: int = 50,
    sr: int = 16000,
) -> list[dict[str, Any]]:
    """
    生成多样化的合成测试集，覆盖常见错误场景。

    场景类型:
      0 - 干净语音 (clean)
      1 - 高噪声 (high_noise)
      2 - 低音量 (low_volume)
      3 - 弱尾音 (weak_tail)
      4 - 爆破音/非语音瞬态 (transient)
      5 - 背景音乐 (music)
    """
    np.random.seed(42)
    test_set = []

    for i in range(num_samples):
        duration = np.random.uniform(2.0, 5.0)
        n_samples = int(duration * sr)
        scene_type = np.random.randint(0, 6)

        # 基础：静音 + 噪声
        if scene_type == 0:
            # 干净语音 — 低噪声
            noise_std = 0.002
            amp = 0.7
        elif scene_type == 1:
            # 高噪声
            noise_std = 0.08
            amp = 0.5
        elif scene_type == 2:
            # 低音量
            noise_std = 0.005
            amp = 0.08
        elif scene_type == 3:
            # 弱尾音 — 语音后半段逐渐衰减
            noise_std = 0.003
            amp = 0.6
        elif scene_type == 4:
            # 非语音瞬态（类似拍手/关门）
            noise_std = 0.003
            amp = 0.0
        else:
            # 背景音乐模拟（低频振荡）
            noise_std = 0.002
            amp = 0.5

        audio = np.random.randn(n_samples).astype(np.float32) * noise_std
        gt_mask = np.zeros(n_samples, dtype=bool)

        # 模拟 1-3 段语音
        num_speech_segments = np.random.randint(1, 4) if scene_type != 4 else 0
        speech_starts = []

        for _ in range(num_speech_segments):
            seg_start = np.random.uniform(0.3, duration - 1.0)
            seg_dur = np.random.uniform(0.3, 1.5)
            seg_end = min(seg_start + seg_dur, duration - 0.1)

            start_idx = int(seg_start * sr)
            end_idx = int(seg_end * sr)
            seg_len = end_idx - start_idx

            # 生成模拟语音信号 (正弦波 + 谐波)
            t = np.arange(seg_len) / sr
            speech_signal = (
                np.sin(2 * np.pi * 200 * t) * 0.5
                + np.sin(2 * np.pi * 400 * t) * 0.3
                + np.sin(2 * np.pi * 800 * t) * 0.15
                + np.random.randn(seg_len).astype(np.float32) * 0.01
            )

            # 应用幅度包络
            envelope = np.ones(seg_len)
            fade_len = min(int(0.05 * sr), seg_len // 4)

            if scene_type == 3:
                # 弱尾音: 后段指数衰减
                envelope[-fade_len:] = np.exp(-3 * np.arange(fade_len) / fade_len)
                envelope[-fade_len:] *= 0.3

            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)

            speech_signal *= envelope

            if scene_type == 2:
                speech_signal *= 0.12  # 低音量

            audio[start_idx:end_idx] += speech_signal * amp * 0.3
            gt_mask[start_idx:end_idx] = True
            speech_starts.append(seg_start)

        # 场景 4: 添加瞬态脉冲
        if scene_type == 4:
            for _ in range(np.random.randint(1, 4)):
                pulse_pos = int(np.random.uniform(0.5, duration - 0.1) * sr)
                pulse_len = int(0.05 * sr)
                pulse = np.sin(2 * np.pi * 1000 * np.arange(pulse_len) / sr)
                pulse *= np.exp(-5 * np.arange(pulse_len) / pulse_len) * 0.5
                end_pos = min(pulse_pos + pulse_len, n_samples)
                audio[pulse_pos:end_pos] += pulse[: end_pos - pulse_pos]

        if scene_type == 5:
            # 添加低频背景音乐
            t_full = np.arange(n_samples) / sr
            music = (
                np.sin(2 * np.pi * 130 * t_full) * 0.1
                + np.sin(2 * np.pi * 260 * t_full) * 0.05
                + np.sin(2 * np.pi * 390 * t_full) * 0.03
            )
            audio += music * 0.15

        # 归一化
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak * 1.1

        test_set.append({
            "audio": audio,
            "mask": gt_mask,
            "scene_type": scene_type,
            "scene_name": ["clean", "high_noise", "low_volume", "weak_tail", "transient", "music"][scene_type],
            "speech_segments": len(speech_starts),
        })

    return test_set


def compute_error_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    """逐类分析错误类型。"""
    # 帧级错误分析
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 边界误差分析
    gt_boundaries = np.diff(gt_mask.astype(int))
    pred_boundaries = np.diff(pred_mask.astype(int))

    onset_errors = np.sum(gt_boundaries == 1) - np.sum(pred_boundaries == 1)
    offset_errors = np.sum(gt_boundaries == -1) - np.sum(pred_boundaries == -1)

    # 连续错误段分析
    from scipy.ndimage import label as nd_label

    fp_segments, n_fp = nd_label(fp.astype(int))
    fn_segments, n_fn = nd_label(fn.astype(int))

    fp_durations = []
    for seg_id in range(1, n_fp + 1):
        fp_durations.append(np.sum(fp_segments == seg_id))

    fn_durations = []
    for seg_id in range(1, n_fn + 1):
        fn_durations.append(np.sum(fn_segments == seg_id))

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm_rate": float(fp / (fp + tn + eps)),
        "miss_rate": float(fn / (fn + tp + eps)),
        "fp_segments": int(n_fp),
        "fn_segments": int(n_fn),
        "fp_duration_mean_ms": float(np.mean(fp_durations) / 16) if fp_durations else 0.0,
        "fn_duration_mean_ms": float(np.mean(fn_durations) / 16) if fn_durations else 0.0,
        "onset_error": int(onset_errors),
        "offset_error": int(offset_errors),
    }


def run_analysis(args: argparse.Namespace) -> None:
    """运行错误模式分析。"""
    print("=" * 70)
    print("VAD 错误模式分析工具")
    print("=" * 70)

    # 生成测试集
    print(f"\n生成 {args.num_samples} 个多样测试样本...")
    test_set = generate_synthetic_test_set(args.num_samples)

    # 初始化 VAD 方法
    vads = {
        "EnergyVAD": EnergyVAD(),
        "SpectralVAD": SpectralVAD(),
        "DNNVAD": DNNVAD(),
    }

    results: dict[str, Any] = {}

    sr = 16000
    for name, vad in vads.items():
        print(f"\n{'─' * 60}")
        print(f"评估: {name}")
        print(f"{'─' * 60}")

        scene_metrics: dict[str, list[dict]] = {s: [] for s in set(s["scene_name"] for s in test_set)}
        all_errors = []

        for sample in test_set:
            audio = sample["audio"]
            gt_mask = sample["mask"]
            scene = sample["scene_name"]

            try:
                segments = vad.detect(audio)
                pred_mask = segments_to_mask(segments, len(audio))
            except Exception as e:
                print(f"  [!] Error on sample: {e}")
                continue

            metrics = compute_error_metrics(pred_mask, gt_mask)
            scene_metrics[scene].append(metrics)
            all_errors.append(metrics)

        # 场景汇总
        print(f"\n  {'Scene':<15s} {'F1':>6s} {'Prec':>6s} {'Recall':>6s} {'FA Rate':>8s} {'Miss':>7s} {'FP Seg':>7s} {'FN Seg':>7s}")
        print(f"  {'─' * 68}")
        for scene in sorted(scene_metrics.keys()):
            m_list = scene_metrics[scene]
            if not m_list:
                continue
            avg = {
                k: np.mean([m[k] for m in m_list])
                for k in ("f1", "precision", "recall", "false_alarm_rate", "miss_rate", "fp_segments", "fn_segments")
            }
            print(
                f"  {scene:<15s} {avg['f1']:>6.4f} {avg['precision']:>6.4f} {avg['recall']:>6.4f} "
                f"{avg['false_alarm_rate']:>8.4f} {avg['miss_rate']:>7.4f} "
                f"{avg['fp_segments']:>7.1f} {avg['fn_segments']:>7.1f}"
            )

        # 总体
        if all_errors:
            overall = {
                k: np.mean([m[k] for m in all_errors])
                for k in ("f1", "precision", "recall", "false_alarm_rate", "miss_rate", "onset_error", "offset_error")
            }
            print(f"  {'─' * 68}")
            print(f"  {'OVERALL':<15s} {overall['f1']:>6.4f} {overall['precision']:>6.4f} {overall['recall']:>6.4f} "
                  f"{overall['false_alarm_rate']:>8.4f} {overall['miss_rate']:>7.4f} "
                  f"{'─':>7s} {'─':>7s}")
            print(f"\n  Onset error (missed onsets): {overall['onset_error']:.1f}")
            print(f"  Offset error (missed offsets): {overall['offset_error']:.1f}")

        results[name] = {
            "scene_metrics": {k: {kk: float(np.mean([m[kk] for m in v])) for kk in v[0]} if v else {} for k, v in scene_metrics.items()},
            "overall": {k: float(v) for k, v in overall.items()},
        }

    # ── 生成分析报告 ──────────────────────────────────────────────────
    if args.generate_report:
        report_path = Path(args.output_dir) / "error_analysis_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "config": {"num_samples": args.num_samples, "sr": sr},
            "vad_methods": results,
            "conclusions": _generate_conclusions(results),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 分析报告已保存: {report_path}")


def _generate_conclusions(results: dict[str, Any]) -> list[str]:
    """根据分析结果生成结论与改进建议。"""
    conclusions = []

    # 找出各方法的最佳和最差场景
    for method, data in results.items():
        scenes = data.get("scene_metrics", {})
        if not scenes:
            continue

        best_scene = max(scenes, key=lambda s: scenes[s].get("f1", 0))
        worst_scene = min(scenes, key=lambda s: scenes[s].get("f1", 0))

        conclusions.append(
            f"{method}: 最佳场景 = '{best_scene}' (F1={scenes[best_scene].get('f1', 0):.4f}), "
            f"最差场景 = '{worst_scene}' (F1={scenes[worst_scene].get('f1', 0):.4f})"
        )

    # 对比建议
    methods = list(results.keys())
    if len(methods) >= 2:
        overall_f1s = {m: results[m]["overall"]["f1"] for m in methods}
        best_method = max(overall_f1s, key=overall_f1s.get)
        conclusions.append(
            f"综合 F1 最优: {best_method} (F1={overall_f1s[best_method]:.4f})"
        )

    conclusions.append("")
    conclusions.append("🔧 改进建议:")
    conclusions.append("  1. 低音量场景：考虑添加自适应增益控制 (AGC) 预处理")
    conclusions.append("  2. 高噪声场景：引入噪声抑制前端或谱减法")
    conclusions.append("  3. 边界偏移：优化 hangover 策略，或使用 CTC/attention 边界预测")
    conclusions.append("  4. 瞬态脉冲误检：加入能量包络连续性约束")
    conclusions.append("  5. 弱尾音漏检：降低尾音判定阈值，或使用 RNN 建模时序依赖")
    conclusions.append("  6. 建议用真实数据替代合成数据进行最终评估")

    return conclusions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD 错误模式分析工具 — 定位模型失败场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/error_analysis.py                           # 运行分析
  python scripts/error_analysis.py --num_samples 100         # 更多样本
  python scripts/error_analysis.py --generate_report         # 生成 JSON 报告
  python scripts/error_analysis.py --output_dir ./analysis   # 指定输出目录
        """,
    )
    parser.add_argument("--num_samples", type=int, default=50, help="测试样本数 (默认: 50)")
    parser.add_argument("--generate_report", action="store_true", help="生成 JSON 分析报告")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="报告输出目录 (默认: ./analysis)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
