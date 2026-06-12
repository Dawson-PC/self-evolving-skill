#!/usr/bin/env bash
#
# run_mcp.sh - MCP服务器启动脚本
#
# 启动 Self-Evolving Skill 的 MCP 服务器，
# 通过 stdin/stdout 与 MCP 客户端通信。
#
# 使用方式:
#   ./run_mcp.sh              # 直接启动
#   ./run_mcp.sh --debug      # 调试模式（输出详细日志）
#   ./run_mcp.sh --help       # 显示帮助
#
# MCP 配置参考:
# 将以下配置添加到 MCP 客户端的配置文件中:
#
# ```json
# {
#   "mcpServers": {
#     "self-evolving-skill": {
#       "command": "/path/to/run_mcp.sh",
#       "args": [],
#       "env": {}
#     }
#   }
# }
# ```
#

set -euo pipefail

# ---- 配置 ----

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Python 可执行文件（优先使用虚拟环境）
PYTHON="${PYTHON:-python3}"

# 存储目录
STORAGE_DIR="${STORAGE_DIR:-$HOME/.openclaw/workspace/self-evolving-skill/storage}"

# 日志级别
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ---- 颜色输出 ----

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- 帮助 ----

show_help() {
    cat <<EOF
Self-Evolving Skill MCP Server

启动 MCP 服务器，通过 stdin/stdout 与 MCP 客户端通信。

用法:
  $(basename "$0") [选项]

选项:
  --debug       启用调试模式（详细日志输出）
  --help        显示此帮助信息

环境变量:
  PYTHON        Python 可执行文件路径 (默认: python3)
  STORAGE_DIR   状态存储目录 (默认: ~/.openclaw/workspace/self-evolving-skill/storage)
  LOG_LEVEL     日志级别 (默认: INFO)

MCP 客户端配置示例:
  {
    "mcpServers": {
      "self-evolving-skill": {
        "command": "$(readlink -f "$0")",
        "args": []
      }
    }
  }
EOF
    exit 0
}

# ---- 参数解析 ----

DEBUG=0
for arg in "$@"; do
    case "$arg" in
        --help)    show_help ;;
        --debug)   DEBUG=1; LOG_LEVEL="DEBUG" ;;
        *)         log_warn "未知参数: $arg" ;;
    esac
done

# ---- 环境检查 ----

# 1. 检查 Python
if ! command -v "$PYTHON" &>/dev/null; then
    log_error "Python 未找到: $PYTHON"
    log_error "请设置 PYTHON 环境变量指向 Python 3.10+ 可执行文件"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1)
log_ok "Python: $PYTHON_VERSION"

# 2. 检查 Python 依赖
MISSING_DEPS=()
for mod in json sys asyncio; do
    if ! "$PYTHON" -c "import $mod" 2>/dev/null; then
        MISSING_DEPS+=("$mod")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    log_warn "缺少 Python 模块: ${MISSING_DEPS[*]}"
    log_info "尝试安装依赖..."
    "$PYTHON" -m pip install --quiet numpy 2>/dev/null || true
fi

# 3. 检查核心模块文件
CORE_DIR="$PROJECT_ROOT/core"
REQUIRED_FILES=(
    "residual_pyramid.py"
    "reflection_trigger.py"
    "experience_replay.py"
    "skill_engine.py"
    "storage.py"
    "mcp_server.py"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CORE_DIR/$f" ]; then
        log_error "核心文件缺失: $CORE_DIR/$f"
        exit 1
    fi
done
log_ok "所有核心模块已就绪"

# 4. 确保存储目录存在
mkdir -p "$STORAGE_DIR"
log_ok "存储目录: $STORAGE_DIR"

# ---- 启动 MCP 服务器 ----

log_info "启动 Self-Evolving Skill MCP Server..."
log_info "核心目录: $CORE_DIR"
log_info "日志级别: $LOG_LEVEL"
log_info "存储目录: $STORAGE_DIR"

if [ $DEBUG -eq 1 ]; then
    log_warn "调试模式已启用"
    export PYTHONDEBUG=1
fi

# 设置 Python 路径以包含项目根目录
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export STORAGE_DIR="$STORAGE_DIR"
export LOG_LEVEL="$LOG_LEVEL"

# 启动 MCP 服务器
# 使用 exec 替换当前进程，确保 stdin/stdout 直通
exec "$PYTHON" -u -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
import logging
logging.basicConfig(
    level=getattr(logging, '$LOG_LEVEL', logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
from core.mcp_server import main
import asyncio
asyncio.run(main())
"
