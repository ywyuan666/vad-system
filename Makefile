# ── VAD System Makefile ──────────────────────────────────────────────
# 用法:
#   make install         安装依赖
#   make train           训练 DNN VAD
#   make test            运行单元测试
#   make benchmark       运行基准评测（含 WebRTC VAD 对比）
#   make export          导出 ONNX
#   make quantize        ONNX INT8 量化
#   make demo            启动 Gradio Web Demo
#   make realtime-demo   启动实时麦克风 VAD Demo
#   make server          启动 FastAPI 生产推理服务
#   make error-analysis  运行 VAD 错误模式分析
#   make docs            查看系统设计文档
#   make lint            运行代码检查 (ruff + mypy)
#   make precommit       安装 pre-commit hooks
#   make docker          构建 Docker 镜像
#   make docker-run      运行 Docker 容器 (Web Demo)
#   make docker-server   运行 Docker 容器 (推理服务)
#   make clean           清理临时文件
# ────────────────────────────────────────────────────────────────────

.PHONY: install train test benchmark export quantize demo realtime-demo server
.PHONY: error-analysis ablation ablation-quick interpret docs lint precommit
.PHONY: docker docker-run docker-server clean

install:
	pip install -r requirements.txt
	pip install "uvicorn[standard]" fastapi websockets prometheus-client python-multipart 2>/dev/null || true
	pip install onnx onnxruntime 2>/dev/null || true
	pip install sounddevice 2>/dev/null || true

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

realtime-demo:
	python demo/realtime_vad.py

server:
	python server/vad_server.py

error-analysis:
	python scripts/error_analysis.py --num_samples 50 --generate_report

ablation:
	python scripts/ablation_study.py --num_samples 30 --output ./ablation

ablation-quick:
	python scripts/ablation_study.py --quick --output ./ablation

interpret:
	python scripts/model_interpretation.py --method all --output ./interpretation

docs:
	@echo "📖 系统设计文档:     docs/system_design.md"
	@echo "📖 Model Card:       docs/MODEL_CARD.md"
	@echo "📖 API 文档:         http://localhost:8000/docs (启动 server 后)"
	@echo "📖 面试 Q&A:         见 docs/system_design.md 第 8 章"
	@echo "📖 消融实验:         make ablation"
	@echo "📖 模型可解释性:     make interpret"

lint:
	ruff check . --line-length=100
	mypy vad/ --ignore-missing-imports --python-version=3.10

precommit:
	pip install pre-commit 2>/dev/null || true
	pre-commit install

docker:
	docker build -t vad-system .

docker-run:
	docker run --rm -p 7860:7860 vad-system

docker-server:
	docker run --rm -p 8000:8000 vad-system python server/vad_server.py

clean:
	rm -rf checkpoints/ results/ __pycache__/ .pytest_cache/ test_out*.txt test_err*.txt
	rm -rf analysis/ ablation/ interpretation/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
