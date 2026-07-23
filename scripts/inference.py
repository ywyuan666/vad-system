#!/usr/bin/env python
"""
VAD 推理脚本
============

支持三种 VAD 方法的推理，输出检测到的语音段。

用法:
    # 能量 VAD（单文件）
    python scripts/inference.py --method energy --audio input.wav

    # 谱 VAD（多文件）
    python scripts/inference.py --method spectral --audio_dir data/test/

    # DNN VAD（加载预训练模型）
    python scripts/inference.py --method dnn --model checkpoints/best.pt --audio input.wav

    # 输出语音段为单独 wav 文件
    python scripts/inference.py --method energy --audio input.wav --export_segments
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from vad import EnergyVAD, SpectralVAD, DNNVAD
from vad.utils import load_audio, save_segments_to_audio


def parse_args():
    parser = argparse.ArgumentParser(description="VAD 推理")
    parser.add_argument("--method", type=str, required=True,
                        choices=["energy", "spectral", "dnn"],
                        help="VAD 方法")
    parser.add_argument("--audio", type=str, help="单个音频文件路径")
    parser.add_argument("--audio_dir", type=str, help="音频目录（批量处理）")

    # VAD 参数
    parser.add_argument("--energy_thresh", type=float, default=None,
                        help="能量阈值（energy 模式，默认自适应）")
    parser.add_argument("--mode", type=str, default="adaptive",
                        choices=["fixed", "adaptive"],
                        help="阈值模式（energy 模式）")
    parser.add_argument("--adaptive_ratio", type=float, default=2.5,
                        help="自适应阈值系数")

    # DNN 参数
    parser.add_argument("--model", type=str, help="DNN 模型路径")
    parser.add_argument("--prob_threshold", type=float, default=0.5,
                        help="DNN 概率阈值")

    # 输出
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--export_segments", action="store_true",
                        help="将每个语音段导出为独立 wav")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出结果")

    return parser.parse_args()


def process_single(audio_path: str, vad, args) -> dict:
    """处理单个音频文件。"""
    audio, sr = load_audio(audio_path)
    segments = vad(audio, sr)

    result = {
        "file": os.path.basename(audio_path),
        "duration_sec": len(audio) / sr,
        "n_segments": len(segments),
        "segments": [
            {"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)}
            for s, e in segments
        ],
    }

    # 导出语音段
    if args.export_segments:
        seg_dir = os.path.join(args.output, "segments", Path(audio_path).stem)
        save_segments_to_audio(audio, sr, segments, seg_dir)
        result["segments_dir"] = seg_dir

    return result


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 创建 VAD 实例
    if args.method == "energy":
        vad = EnergyVAD(
            mode=args.mode,
            energy_thresh=args.energy_thresh,
            adaptive_ratio=args.adaptive_ratio,
        )
    elif args.method == "spectral":
        vad = SpectralVAD()
    elif args.method == "dnn":
        if not args.model:
            print("错误: dnn 模式需要 --model 参数")
            sys.exit(1)
        vad = DNNVAD(model_path=args.model, prob_threshold=args.prob_threshold)
    else:
        raise ValueError(f"未知方法: {args.method}")

    # 收集待处理文件
    if args.audio:
        audio_paths = [args.audio]
    elif args.audio_dir:
        audio_paths = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
            audio_paths.extend(str(p) for p in Path(args.audio_dir).rglob(ext))
        if not audio_paths:
            print(f"错误: 目录 {args.audio_dir} 中未找到音频文件")
            sys.exit(1)
    else:
        print("错误: 请指定 --audio 或 --audio_dir")
        sys.exit(1)

    # 批量处理
    all_results = []
    for path in audio_paths:
        print(f"处理: {path}")
        try:
            result = process_single(path, vad, args)
            all_results.append(result)

            if not args.json:
                print(f"  时长: {result['duration_sec']:.1f}s | "
                      f"检测到 {result['n_segments']} 个语音段")
                for seg in result["segments"]:
                    print(f"    [{seg['start']:.2f}s - {seg['end']:.2f}s] "
                          f"({seg['duration']:.2f}s)")
        except Exception as e:
            print(f"  处理失败: {e}")

    # 汇总
    total_duration = sum(r["duration_sec"] for r in all_results)
    total_speech = sum(
        sum(s["duration"] for s in r["segments"]) for r in all_results
    )
    avg_speech_ratio = total_speech / total_duration if total_duration > 0 else 0
    print(f"\n汇总:")
    print(f"  总时长: {total_duration:.1f}s | "
          f"语音占比: {avg_speech_ratio:.1%} | "
          f"文件数: {len(all_results)}")

    # 保存结果
    result_path = os.path.join(args.output, "vad_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果保存: {result_path}")


if __name__ == "__main__":
    main()
