#!/bin/bash
# ── VAD Docker 入口脚本 ──────────────────────────────────────────
# 支持多种运行模式:
#   web       - 启动 Gradio Web Demo (默认)
#   server    - 启动 FastAPI 推理服务
#   train     - 训练 DNN VAD
#   eval      - 运行基准评测 + 错误分析
#   shell     - 进入交互式 Shell
# ─────────────────────────────────────────────────────────────────

set -e

MODE="${1:-web}"

case "$MODE" in
    web)
        echo "▶ 启动 Gradio Web Demo (端口 7860)"
        exec python demo/app.py
        ;;
    server)
        echo "▶ 启动 FastAPI 推理服务 (端口 8000)"
        exec python server/vad_server.py
        ;;
    train)
        echo "▶ 训练 DNN VAD (合成数据, 30 epochs)"
        python scripts/train.py --method synthetic --epochs 30 --output_dir checkpoints
        echo "✅ 训练完成"
        ;;
    eval)
        echo "▶ 运行基准评测..."
        python scripts/train.py --method synthetic --epochs 10 --output_dir checkpoints
        python benchmark/benchmark.py --method synthetic --n_samples 50 --dnn_model checkpoints/best.pt --output results
        python scripts/error_analysis.py --num_samples 30 --generate_report --output_dir analysis
        echo "✅ 评测完成: 结果在 results/ 和 analysis/"
        ;;
    shell)
        echo "▶ 进入交互式 Shell"
        exec /bin/bash
        ;;
    *)
        echo "用法: docker run vad-system [web|server|train|eval|shell]"
        echo "  默认: web"
        exit 1
        ;;
esac
