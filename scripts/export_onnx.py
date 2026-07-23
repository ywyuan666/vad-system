#!/usr/bin/env python
"""
ONNX 模型导出脚本
==================

将训练好的 DNN VAD 模型 (PyTorch .pt) 导出为 ONNX 格式，
可用于 TensorRT / ONNX Runtime 等推理框架进行生产部署。

用法:
    python scripts/export_onnx.py --model checkpoints/best.pt \\
                                  --output checkpoints/best.onnx \\
                                  --n_mels 40 --seq_len 200

特性:
    - 动态 batch 和序列长度 (opset 17)
    - 支持静态/动态推理
    - 导出后自动验证精度一致性
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from vad.dnn_vad import VADNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNN VAD ONNX 导出")
    parser.add_argument("--model", type=str, required=True, help="PyTorch 模型路径 (.pt)")
    parser.add_argument("--output", type=str, default=None, help="输出路径 (.onnx)")
    parser.add_argument("--n_mels", type=int, default=40, help="Fbank 维度")
    parser.add_argument("--seq_len", type=int, default=200, help="导出时固定的序列长度（帧数）")
    parser.add_argument("--dynamic_batch", action="store_true", default=True, help="动态 batch 维度")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--verify", action="store_true", default=True, help="导出后验证精度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        args.output = args.model.replace(".pt", ".onnx")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1. 加载模型
    print(f"[1/4] 加载模型: {args.model}")
    device = torch.device("cpu")
    model = VADNet(n_mels=args.n_mels).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=True)
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {total_params:,} ({total_params/1000:.1f}K)")

    # 2. 构建 dummy 输入
    print(f"[2/4] 构建 dummy 输入: (1, {args.seq_len}, {args.n_mels})")
    dummy_input = torch.randn(1, args.seq_len, args.n_mels)

    # 3. 导出 ONNX
    print(f"[3/4] 导出 ONNX -> {args.output}")
    print(f"      opset: {args.opset}, dynamic_batch: {args.dynamic_batch}")

    dynamic_axes = (
        {
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"},
        }
        if args.dynamic_batch
        else None
    )

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    file_size = os.path.getsize(args.output)
    print(f"      文件大小: {file_size / 1024:.1f} KB")
    print(f"      ONNX 导出成功！")

    # 4. 验证精度一致性
    if args.verify:
        print(f"[4/4] 验证精度一致性...")
        try:
            import onnx
            import onnxruntime

            # 检查 ONNX 模型
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print(f"      ONNX 模型结构检查通过")

            # 对比推理结果
            ort_session = onnxruntime.InferenceSession(args.output)

            with torch.no_grad():
                pt_output = model(dummy_input).numpy()

            ort_input = dummy_input.numpy().astype(np.float32)
            ort_output = ort_session.run(["output"], {"input": ort_input})[0]

            max_diff = np.max(np.abs(pt_output - ort_output))
            print(f"      PyTorch vs ONNX Runtime 最大差异: {max_diff:.2e}")

            if max_diff < 1e-4:
                print(f"      ✅ 精度验证通过！ONNX 模型可直接用于生产部署。")
            else:
                print(f"      ⚠️  精度差异偏大 ({max_diff:.2e})，建议检查导出参数。")
        except ImportError:
            print(f"      [SKIP] 需要 onnx 和 onnxruntime 库进行验证")
            print(f"      pip install onnx onnxruntime")

    print(f"\n完成！ONNX 模型: {args.output}")


if __name__ == "__main__":
    main()
