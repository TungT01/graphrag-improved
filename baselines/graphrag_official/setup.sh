#!/bin/bash
# baselines/graphrag_official/setup.sh
# ------------------------------------
# 安装 microsoft/graphrag 并初始化工作目录。
#
# 用法：
#   cd baselines/graphrag_official
#   bash setup.sh
#
# 注意：
#   1. 需要 Python 3.10+（graphrag 要求）
#   2. 安装完成后，需要在 graphrag_workspace/settings.yaml 中配置 LLM API key
#   3. 推荐使用虚拟环境：python -m venv .venv && source .venv/bin/activate

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/graphrag_workspace"

echo "============================================================"
echo "  Microsoft GraphRAG 官方基线 - 安装脚本"
echo "============================================================"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Python 版本：$PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "警告：graphrag 需要 Python 3.10+，当前版本为 $PYTHON_VERSION"
    echo "建议使用 pyenv 或 conda 安装 Python 3.10+"
    echo "继续安装，但可能遇到兼容性问题..."
fi

# 安装 graphrag
echo ""
echo "步骤 1/3：安装 graphrag..."
pip install "graphrag>=0.3.0" --quiet
echo "  ✓ graphrag 安装完成"

# 安装其他依赖
echo ""
echo "步骤 2/3：安装其他依赖..."
pip install "tiktoken>=0.5.0" --quiet
echo "  ✓ 依赖安装完成"

# 初始化工作目录
echo ""
echo "步骤 3/3：初始化 GraphRAG 工作目录..."
mkdir -p "${WORKSPACE_DIR}/input"

cd "${WORKSPACE_DIR}"

# 初始化 graphrag 配置
if [ ! -f "${WORKSPACE_DIR}/settings.yaml" ]; then
    python3 -m graphrag init --root . 2>/dev/null || {
        echo "  [提示] graphrag init 命令失败，尝试手动创建配置..."
        # 手动创建最小配置
        cat > "${WORKSPACE_DIR}/settings.yaml" << 'EOF'
# GraphRAG 配置文件
# 请填写您的 LLM API key

encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true
  max_tokens: 4000
  temperature: 0
  top_p: 1
  n: 1
  request_timeout: 180.0
  api_base: null
  api_version: null
  organization: null
  proxy: null
  cognitive_services_endpoint: null
  deployment_name: null
  tokens_per_minute: 0
  requests_per_minute: 0
  max_retries: 10
  max_retry_wait: 10.0
  sleep_on_rate_limit_recommendation: true
  concurrent_requests: 25

parallelization:
  stagger: 0.3
  num_threads: 50

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 25
    batch_size: 16
    batch_max_tokens: 8191
    target: required

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "output/reports"

entity_extraction:
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

summarize_descriptions:
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  enabled: false

community_reports:
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 32
EOF
        echo "  ✓ 手动创建 settings.yaml"
    }
else
    echo "  settings.yaml 已存在，跳过初始化"
fi

echo ""
echo "============================================================"
echo "  安装完成！"
echo "============================================================"
echo ""
echo "后续步骤："
echo "  1. 设置 API key："
echo "     export GRAPHRAG_API_KEY='your-openai-api-key'"
echo "     或编辑 ${WORKSPACE_DIR}/settings.yaml"
echo ""
echo "  2. 将文档放入 input 目录："
echo "     cp /path/to/your/docs/*.txt ${WORKSPACE_DIR}/input/"
echo ""
echo "  3. 运行索引："
echo "     python indexer.py --input_dir /path/to/docs"
echo ""
echo "  4. 运行评估："
echo "     python evaluator.py --data_dir /path/to/multihop_rag --search_type local"
echo ""
