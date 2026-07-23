#!/usr/bin/env python
"""
预训练模型下载脚本
===================

从 HuggingFace / GitHub Releases 下载预训练的 DNN VAD 模型。
首次使用建议直接本地训练，更快:

    python scripts/train.py --method synthetic --epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预训练 VAD 模型下载")
    parser.add_argument("--source", type=str, default="huggingface",
                        choices=["huggingface", "github", "manual"],
                        help="下载来源")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="保存目录")
    return parser.parse_args()


def download_huggingface(output_dir: str) -> str:
    model_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(model_path):
        print(f"模型已存在: {model_path}")
        return model_path
    os.makedirs(output_dir, exist_ok=True)
    hf_url = "https://huggingface.co/ywyuan666/vad-system/resolve/main/best.pt"
    print(f"从 HuggingFace 下载: {hf_url}")
    try:
        import requests
        resp = requests.get(hf_url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(model_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    print(f"\r  进度: {downloaded/total*100:.0f}%", end="", flush=True)
        print(f"\n完成: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")
        return model_path
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("提示: 可自行训练 python scripts/train.py --method synthetic --epochs 30")
        return ""


def main() -> None:
    args = parse_args()
    print("VAD 预训练模型下载")
    print("=" * 40)
    if args.source == "huggingface":
        download_huggingface(args.output)
    elif args.source == "github":
        print("请使用 HuggingFace 或直接本地训练")
    else:
        print("手动下载后放入 checkpoints/ 目录即可")


if __name__ == "__main__":
    main()
