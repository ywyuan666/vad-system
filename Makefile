# ── VAD System Makefile ──────────────────────────────────────────────
# 用法:
#   make install         安装依赖
#   make train           训练 DNN VAD
#   make test            运行单元测试
#   make benchmark       运行基准评测（含 WebRTC VAD 对比）
#   make export          导出 ONNX
#   make quantize        ONNX INT8 量化
#   make demo            启动 Web Demo
#   make docker          构建 Docker 镜像
#   make docker-run      运行 Docker 容器
#   make clean           清理临时文件
# ────────────────────────────────────────────────────────────────────

.PHONY: install train test benchmark export quantize demo docker docker-run clean

install:
	pip install -r requirements.txt
	pip install onnx onnxruntime 2>/dev/null || true

train:
	python scripts/train.py --method synthetic --epochs 30 --output_dir checkpoints

test:
	python -m pytest tests/ -v --tb=short

benchmark:
	python benchmark/benchmark.py --method synthetic --n_samples 100 --dnn_model checkpoints/best.pt --output results

export:
	python scripts/export_onnx.py --model checkpoints/best.pt --output checkpoints/best.onnx

quantize:
	python scripts/quantize_onnx.py --input checkpoints/best.onnx --output checkpoints/best_int8.onnx

demo:
	python demo/app.py

docker:
	docker build -t vad-system .

docker-run:
	docker run --rm -p 7860:7860 vad-system

clean:
	rm -rf checkpoints/ results/ __pycache__/ .pytest_cache/ test_out*.txt test_err*.txt
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
