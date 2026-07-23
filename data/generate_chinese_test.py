#!/usr/bin/env python3
"""
中文语音测试数据生成
======================
生成包含中文语音的测试音频，用于验证 VAD 在中文场景下的表现。

中文语音特点 vs 英文:
  - 音节边界更清晰 (单音节字)
  - 四声声调 (频率变化更剧烈)
  - 语速通常较快 (约 3-4 字/秒)
  - 轻声/儿化音能量更低

用法:
  python data/generate_chinese_test.py                       # 生成全部
  python data/generate_chinese_test.py --num_samples 20       # 指定数量
  python data/generate_chinese_test.py --output ./data/chinese_test
  python data/generate_chinese_test.py --eval                 # 生成后自动评测
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vad import EnergyVAD, SpectralVAD, DNNVAD, VADEvaluator
from vad.utils import segments_to_mask, ensure_sr


# ── 中文语音模拟 ──────────────────────────────────────────────────────────

# 常见中文音节: 拼音 + 基频范围
CHINESE_SYLLABLES = [
    ("nǐ", 220, 280),   # 你 (上声: 先降后升)
    ("hǎo", 200, 260),  # 好 (上声)
    ("wǒ", 210, 270),   # 我 (上声)
    ("shì", 180, 220),  # 是 (去声: 先平后降)
    ("zhōng", 200, 240),  # 中 (阴平: 高平)
    ("guó", 190, 250),  # 国 (阳平: 上升)
    ("rén", 180, 230),  # 人 (阳平)
    ("dà", 170, 210),   # 大 (去声)
    ("xiǎo", 220, 280), # 小 (上声)
    ("xué", 200, 260),  # 学 (阳平)
    ("shēng", 190, 230), # 生 (阴平)
    ("tiān", 200, 240), # 天 (阴平)
    ("qì", 180, 220),   # 气 (去声)
    ("shuǐ", 210, 270), # 水 (上声)
    ("huǒ", 200, 260),  # 火 (上声)
    ("shān", 180, 230), # 山 (阴平)
    ("hǎi", 200, 260),  # 海 (上声)
    ("fēng", 190, 230), # 风 (阴平)
    ("yǔ", 200, 250),   # 雨 (上声)
    ("qíng", 190, 240), # 晴 (阳平)
]

# 中文常用短句 (用于生成连续语音)
CHINESE_PHRASES = [
    [("nǐ", "hǎo"), ("shì", "jiè")],      # 你好世界
    [("zhōng", "guó"), ("rén", "mín")],    # 中国人民
    [("dà", "xué"), ("xué", "shēng")],     # 大学生
    [("tiān", "qì"), ("qíng", "lǎng")],    # 天气晴朗
    [("shān", "shuǐ"), ("fēng", "guāng")], # 山水风光
]


def synthesize_chinese_syllable(
    syllable: tuple[str, float, float],
    duration: float = 0.25,
    sr: int = 16000,
) -> np.ndarray:
    """
    合成一个中文音节。

    模拟中文发音特点:
      - 基频包络: 根据声调变化 (上声: 先降后升; 去声: 下降; 阳平: 上升; 阴平: 平直)
      - 谐波结构: 模拟声道共振
      - 能量包络: 音节起始和结束的平滑过渡
    """
    n = int(duration * sr)
    t = np.arange(n) / sr
    name, f0_start, f0_end = syllable

    # 根据起止基频差异判断声调类型
    f0_diff = f0_end - f0_start

    if f0_diff > 20:
        # 阳平 / 上声 (频率上升)
        f0 = f0_start + (f0_end - f0_start) * t / duration
    elif f0_diff < -20:
        # 去声 (频率下降)
        f0 = f0_start + (f0_end - f0_start) * t / duration
    else:
        # 阴平 (频率基本不变)
        f0 = np.ones_like(t) * f0_start

    # 基频 + 3 次谐波
    signal = (
        np.sin(2 * np.pi * f0 * t) * 0.5         # 基频
        + np.sin(2 * np.pi * f0 * 2 * t) * 0.25  # 二次谐波
        + np.sin(2 * np.pi * f0 * 3 * t) * 0.12  # 三次谐波
        + np.sin(2 * np.pi * f0 * 4 * t) * 0.06  # 四次谐波
    )

    # 能量包络 (模拟声门脉冲)
    fade_len = int(0.03 * sr)
    envelope = np.ones(n)
    envelope[:fade_len] = np.linspace(0, 1, fade_len) ** 0.7
    envelope[-fade_len:] = np.linspace(1, 0, fade_len) ** 1.3

    signal *= envelope

    # 添加微小的频率抖动 (模拟自然发音的不稳定性)
    jitter = np.sin(2 * np.pi * 5 * t) * 2.0  # 5Hz 抖动, ±2Hz
    signal = np.sin(2 * np.pi * (f0 + jitter) * t) * 0.5 * envelope

    return signal.astype(np.float32)


def generate_chinese_audio(
    num_phrases: int = 3,
    sr: int = 16000,
    noise_level: str = "clean",
) -> tuple[np.ndarray, list[tuple[float, float]], list[str]]:
    """
    生成包含中文语音的测试音频。

    Args:
        num_phrases: 使用的词组数量
        sr: 采样率
        noise_level: 噪声水平 (clean/noisy/low_vol)

    Returns:
        (audio, segments, transcripts)
        segments: [(start_sec, end_sec), ...]
        transcripts: 每段对应的中文文本
    """
    if noise_level == "clean":
        noise_std, amp = 0.002, 0.7
    elif noise_level == "noisy":
        noise_std, amp = 0.05, 0.5
    else:
        noise_std, amp = 0.003, 0.08

    # 选择词组
    selected = CHINESE_PHRASES[:num_phrases]
    if num_phrases > len(CHINESE_PHRASES):
        # 随机组合
        for _ in range(num_phrases - len(CHINESE_PHRASES)):
            idx = np.random.randint(0, len(CHINESE_PHRASES))
            selected.append(CHINESE_PHRASES[idx])

    # 构建音频
    silence_gaps = np.random.uniform(0.4, 1.0, num_phrases)  # 语素间停顿
    total_duration = sum(silence_gaps) + num_phrases * 0.6  # 每词组 ~0.6s
    total_samples = int(total_duration * sr)

    audio = np.random.randn(total_samples).astype(np.float32) * noise_std
    segments = []
    transcripts = []
    current_pos = 0

    for i, phrase in enumerate(selected):
        phrase_start = current_pos + silence_gaps[i] * 0.3 * sr
        phrase_end = phrase_start
        phrase_text = ""

        for syllable in phrase:
            # 在音节字典中查找
            syllable_data = None
            for s in CHINESE_SYLLABLES:
                if s[0] == syllable[0]:
                    syllable_data = s
                    break

            if syllable_data is None:
                # 近似
                syllable_data = (syllable[0], 200, 250)

            syl_dur = 0.15 + np.random.uniform(0, 0.08)
            signal = synthesize_chinese_syllable(syllable_data, syl_dur, sr)

            end_pos = int(phrase_end + len(signal))
            if end_pos > len(audio):
                break

            audio[phrase_end:end_pos] += signal * amp * 0.3
            phrase_end = end_pos
            phrase_text += syllable[0]

            # 音节间微停顿
            phrase_end += int(0.02 * sr)

        seg_start = phrase_start / sr
        seg_end = min(phrase_end, total_samples) / sr
        segments.append((seg_start, seg_end))
        transcripts.append(phrase_text)

        current_pos = int(phrase_end) + int(silence_gaps[i] * 0.7 * sr)

    # 归一化
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak * 1.1

    return audio, segments, transcripts


def run_evaluation(args: argparse.Namespace) -> None:
    """在中文语音上评测各 VAD 方法。"""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  中文语音 VAD 评测")
    print("=" * 60)

    scenarios = ["clean", "noisy", "low_vol"]
    methods = {
        "EnergyVAD": EnergyVAD(),
        "SpectralVAD": SpectralVAD(),
        "DNNVAD": DNNVAD(),
    }

    all_results: list[dict] = []

    for noise in scenarios:
        print(f"\n{'─' * 40}")
        print(f"场景: {noise.upper()}")
        print(f"{'─' * 40}")

        audio, segments_gt, transcripts = generate_chinese_audio(
            num_phrases=args.num_samples,
            noise_level=noise,
        )

        gt_mask = segments_to_mask(segments_gt, len(audio))
        print(f"  词组: {transcripts}")
        print(f"  标注段: {segments_gt}")
        print(f"  音频长度: {len(audio)/16000:.2f}s")

        for name, vad in methods.items():
            try:
                segs_pred = vad.detect(audio)
                pred_mask = segments_to_mask(segs_pred, len(audio))
                evaluator = VADEvaluator(pred_mask, gt_mask)
                metrics = evaluator.compute_frame_metrics()
                all_results.append({
                    "method": name,
                    "scenario": noise,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "far": metrics.get("false_alarm_rate", 0),
                    "miss": metrics.get("miss_rate", 0),
                })
                print(f"  {name:<15s}: F1={metrics['f1']:.4f}  "
                      f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
            except Exception as e:
                print(f"  {name:<15s}: ❌ {e}")

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  中文 VAD 评测汇总")
    print(f"{'=' * 60}")
    print(f"  {'Method':<15s} {'Scenario':<10s} {'F1':>8s} {'Prec':>7s} {'Recall':>7s} {'FAR':>7s} {'Miss':>7s}")
    print(f"  {'─' * 62}")
    for r in all_results:
        print(f"  {r['method']:<15s} {r['scenario']:<10s} "
              f"{r['f1']:>8.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['far']:>7.4f} {r['miss']:>7.4f}")

    # 保存
    report_path = output_dir / "chinese_vad_eval.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": args.num_samples,
            "methods": list(methods.keys()),
            "scenarios": scenarios,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评测结果保存: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="中文语音 VAD 测试数据生成与评测",
    )
    parser.add_argument("--num_samples", type=int, default=3,
                        help="词组数量 (默认: 3)")
    parser.add_argument("--output", type=str, default="./data/chinese_test",
                        help="输出目录 (默认: ./data/chinese_test)")
    parser.add_argument("--eval", action="store_true",
                        help="生成后自动运行评测")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        run_evaluation(args)
    else:
        # 仅生成示例
        audio, segments, transcripts = generate_chinese_audio(
            num_phrases=args.num_samples, noise_level="clean"
        )
        print("✅ 中文语音测试数据生成:")
        for i, (s, e) in enumerate(segments):
            print(f"  [{s:.2f}s - {e:.2f}s] {transcripts[i]}")
        print(f"  总时长: {len(audio)/16000:.2f}s")

        # 保存参考标注
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
            json.dump({
                "segments": [{"start": s, "end": e, "text": t}
                             for (s, e), t in zip(segments, transcripts)],
                "sr": 16000,
            }, f, indent=2, ensure_ascii=False)
        print(f"  标注保存: {output_dir / 'reference.json'}")
