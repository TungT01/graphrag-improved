#!/bin/bash
# ============================================================
# GraphRAG 官方基线 - 20 篇文章小规模测试启动脚本
# LLM：Kimi API（moonshot-v1-32k）
# Embedding：本地 sentence-transformers/all-MiniLM-L6-v2
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv-graphrag/bin/python"
WORKSPACE="$SCRIPT_DIR/graphrag_workspace"
CORPUS_JSON="/Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json"
EMBEDDING_SERVER_PID=""

# 清理函数：脚本退出时关闭 embedding 服务
cleanup() {
    if [ -n "$EMBEDDING_SERVER_PID" ]; then
        echo ""
        echo "正在关闭本地 embedding 服务（PID: $EMBEDDING_SERVER_PID）..."
        kill "$EMBEDDING_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "============================================================"
echo "GraphRAG 官方基线 - 小规模测试（20 篇文章）"
echo "  LLM      : Kimi moonshot-v1-32k"
echo "  Embedding: 本地 all-MiniLM-L6-v2（端口 11434）"
echo "============================================================"

# ── 检查 API Key ──────────────────────────────────────────────
if grep -E "^GRAPHRAG_API_KEY=YOUR_API_KEY_HERE" "$WORKSPACE/.env" > /dev/null 2>&1; then
    echo ""
    echo "❌ 错误：请先在以下文件中填入你的 Kimi API Key："
    echo "   $WORKSPACE/.env"
    exit 1
fi
export $(grep -v '^#' "$WORKSPACE/.env" | xargs)
echo "✓ API Key 已配置"

# ── 步骤 1：启动本地 embedding 服务 ──────────────────────────
echo ""
echo "步骤 1/4：启动本地 embedding 服务..."

# 检查端口是否已被占用
if curl -s http://localhost:11434/health > /dev/null 2>&1; then
    echo "  ✓ embedding 服务已在运行（端口 11434）"
else
    python3 "$SCRIPT_DIR/local_embedding_server.py" &
    EMBEDDING_SERVER_PID=$!
    echo "  启动中（PID: $EMBEDDING_SERVER_PID），等待就绪..."

    # 最多等 30 秒
    for i in $(seq 1 30); do
        sleep 1
        if curl -s http://localhost:11434/health > /dev/null 2>&1; then
            echo "  ✓ embedding 服务已就绪（${i}s）"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "  ❌ embedding 服务启动超时，请检查日志"
            exit 1
        fi
    done
fi

# ── 步骤 2：准备 20 篇文章的输入文件 ─────────────────────────
echo ""
echo "步骤 2/4：准备输入文档（取前 20 篇）..."
mkdir -p "$WORKSPACE/input"

$VENV_PYTHON - "$WORKSPACE/input" "$CORPUS_JSON" <<'PYEOF'
import json, os, sys

input_dir = sys.argv[1]
corpus_path = sys.argv[2]

with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# 清空旧文件
for f in os.listdir(input_dir):
    if f.endswith(".txt"):
        os.remove(os.path.join(input_dir, f))

# 取前 20 篇
docs = corpus[:20]
for i, doc in enumerate(docs):
    title = doc.get("title", f"doc_{i}").replace("/", "_").replace("\\", "_")
    body = doc.get("body", "")
    filename = f"{i:03d}_{title[:50]}.txt"
    filepath = os.path.join(input_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n\n{body}")

print(f"✓ 已写入 {len(docs)} 篇文章到 {input_dir}")
PYEOF

# ── 步骤 3：运行 graphrag index ───────────────────────────────
echo ""
echo "步骤 3/4：运行 GraphRAG 索引构建..."
echo "  预计耗时：10-20 分钟（取决于 Kimi API 速率）"
echo ""

cd "$WORKSPACE"
$VENV_PYTHON -m graphrag index --root .

# ── 步骤 4：验证输出 ──────────────────────────────────────────
echo ""
echo "步骤 4/4：验证索引输出..."
if [ -d "$WORKSPACE/output" ] && ls "$WORKSPACE/output"/*.parquet 2>/dev/null | head -1 > /dev/null; then
    echo "✓ 索引构建完成！输出文件："
    ls "$WORKSPACE/output/"*.parquet 2>/dev/null | head -10
    echo ""
    echo "============================================================"
    echo "✓ GraphRAG 官方基线索引构建成功！"
    echo "  下一步运行评估："
    echo "  python $SCRIPT_DIR/evaluator.py --root $WORKSPACE --mode local"
    echo "============================================================"
else
    echo "⚠️  未找到 .parquet 输出文件，请检查 output/logs 目录"
    ls "$WORKSPACE/output/" 2>/dev/null || true
fi
