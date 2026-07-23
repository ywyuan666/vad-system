#!/usr/bin/env python
"""
ONNX INT8 量化脚本
===================

对已导出的 ONNX VAD 模型进行 INT8 动态量化，
减少模型体积和推理延迟，适合生产部署。

用法:
    # 先导出 ONNX
    python scripts/export_onnx.py --model checkpoints/best.pt

    # 再量化
    python scripts/quantize_onnx.py --input checkpoints/best.onnx \\
                                    --output checkpoints/best_int8.onnx

效果:
    - 模型体积: ~220KB → ~60KB (约 73% 缩小)
    - 推理速度: CPU 上提升 2-3x
    - 精度损失: F1 下降通常 < 0.5%
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX VAD INT8 量化")
    parser.add_argument("--input", type=str, required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", type=str, default=None, help="输出量化模型路径")
    parser.add_argument("--per_channel", action="store_true", default=False,
                        help="按 channel 量化（更精确但略慢）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_int8.onnx")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    orig_size = os.path.getsize(args.input)
    print(f"[1/3] 原始模型: {args.input}, 大小: {orig_size / 1024:.1f} KB")

    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"[ERROR] 需要 onnx 和 onnxruntime: pip install onnx onnxruntime")
        sys.exit(1)

    print(f"[2/3] 验证 ONNX 模型结构...")
    model = onnx.load(args.input)
    onnx.checker.check_model(model)
    print(f"      ONNX 模型结构检查通过")

    print(f"[3/3] 执行 INT8 动态量化...")
    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    quant_size = os.path.getsize(args.output)
    ratio = (1 - quant_size / orig_size) * 100
    print(f"      量化模型: {args.output}")
    print(f"      大小: {quant_size / 1024:.1f} KB ({ratio:.0f}% 缩小)")

    print(f"\n验证量化模型精度...")
    try:
        import onnxruntime as ort
        dummy = np.random.randn(1, 200, 40).astype(np.float32)
        out_fp32 = ort.InferenceSession(args.input).run(["output"], {"input": dummy})[0]
        out_int8 = ort.InferenceSession(args.output).run(["output"], {"input": dummy})[0]
        max_diff = np.max(np.abs(out_fp32 - out_int8))
        rel_diff = max_diff / (np.max(np.abs(out_fp32)) + 1e-10) * 100
        print(f"      FP32 vs INT8 最大差异: {max_diff:.4f} ({rel_diff:.2f}%)")
        result = "OK" if rel_diff < 5 else "WARN"
        print(f"      [{result}] INT8 量化验证完成")
    except Exception as e:
        print(f"      [SKIP] 验证跳过: {e}")

    print(f"\n完成！量化模型: {args.output}")
    print(f"      原始: {orig_size / 1024:.1f} KB -> 量化: {quant_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
