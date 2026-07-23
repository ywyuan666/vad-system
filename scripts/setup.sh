#!/usr/bin/env bash
# =============================================================================
# VAD 系统一键环境配置脚本
# =============================================================================
#
# 用法:
#   source scripts/setup.sh           # 自动检测 GPU + 创建 venv + 安装依赖
#   source scripts/setup.sh --gpu     # 强制 GPU 模式
#   source scripts/setup.sh --cpu     # 强制 CPU 模式
#
# 提供公共函数供其他脚本 source 使用:
#   cd_project          — 切换到项目根目录
#   detect_gpu          — 检测 GPU 可用性
#   ensure_venv         — 确保虚拟环境已激活
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色定义
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# 公共函数（可被其他脚本 source 使用）
# =============================================================================

cd_project() {
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
}

detect_gpu() {
    # 检测 NVIDIA GPU 可用性
    # 返回值: 0=GPU可用, 1=GPU不可用
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

ensure_venv() {
    # 确保虚拟环境存在并激活
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        log_info "已在虚拟环境中: $VIRTUAL_ENV"
        return 0
    fi

    local venv_dir="$PROJECT_ROOT/.venv"
    if [ ! -d "$venv_dir" ]; then
        log_info "创建虚拟环境: $venv_dir"
        python3 -m venv "$venv_dir"
        log_ok "虚拟环境已创建"
    fi

    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    log_ok "已激活虚拟环境: $venv_dir"
}

# =============================================================================
# 如果直接执行（非 source），运行一键配置
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    log_info "VAD 系统一键环境配置"
    log_info "项目目录: $PROJECT_ROOT"
    echo ""

    # 1. 切换到项目目录
    cd_project
    log_ok "工作目录: $(pwd)"

    # 2. 检测 GPU
    echo ""
    log_info "检测 GPU..."
    if detect_gpu; then
        GPU_AVAILABLE=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        log_ok "检测到 GPU: $GPU_NAME"
    else
        GPU_AVAILABLE=false
        log_warn "未检测到 NVIDIA GPU，将使用 CPU"
    fi

    # 3. 创建并激活虚拟环境
    echo ""
    ensure_venv

    # 4. 升级 pip
    echo ""
    log_info "升级 pip..."
    pip install --upgrade pip -q
    log_ok "pip 已升级"

    # 5. 安装依赖
    echo ""
    log_info "安装 Python 依赖..."
    pip install -r "$PROJECT_ROOT/requirements.txt" -q

    # 根据 GPU 情况安装 PyTorch
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "安装 GPU 版 PyTorch..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    else
        log_info "安装 CPU 版 PyTorch..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi
    log_ok "依赖安装完成"

    # 6. 创建必要目录
    mkdir -p "$PROJECT_ROOT/checkpoints"
    mkdir -p "$PROJECT_ROOT/results"

    echo ""
    log_ok "配置完成！"
    echo ""
    echo "  使用方式:"
    echo "    python scripts/train.py --method synthetic    # 训练 DNN VAD"
    echo "    python scripts/inference.py --method energy   # 推理"
    echo "    python demo/app.py                            # 启动 Web Demo"
    echo ""
fi
