# GraphRAG Improved — 基线系统

本目录包含两个对比实验基线系统，用于与 GraphRAG Improved（结构熵惩罚 Leiden + U-Retrieval）进行性能对比。

## 目录结构

```
baselines/
├── naive_rag/                    # Naive RAG 基线
│   ├── indexer.py                # 文档索引构建（句子级向量化）
│   ├── retriever.py              # 检索器（余弦相似度 Top-K）
│   ├── evaluator.py              # 评估器（P@K, R@K, MRR, NDCG, ROUGE-L）
│   └── requirements.txt          # 依赖列表
│
├── graphrag_official/            # Microsoft GraphRAG 官方基线
│   ├── setup.sh                  # 安装脚本
│   ├── indexer.py                # 文档索引构建（调用 graphrag index）
│   ├── retriever.py              # 检索器（local/global search）
│   ├── evaluator.py              # 评估器
│   └── requirements.txt          # 依赖列表
│
├── data_loader.py                # MultiHop-RAG 数据集加载器
├── run_evaluation.py             # 统一评估入口
└── README.md                     # 本文件
```

## 数据集

本项目使用 **MultiHop-RAG** 数据集进行评估。

**数据集已预置在：**
- `/Users/ttung/Desktop/个人学习/data/multihop_rag/`（corpus.json + MultiHopRAG.json）
- `/Users/ttung/Desktop/个人学习/data/MultiHop-RAG/dataset/`（原始 GitHub 克隆）

**如需重新下载：**
```bash
# 方式一：GitHub
git clone https://github.com/yixuantt/MultiHop-RAG
cp -r MultiHop-RAG/dataset/ /Users/ttung/Desktop/个人学习/data/multihop_rag/

# 方式二：HuggingFace
huggingface-cli download yixuantt/MultiHopRAG --repo-type dataset
```

---

## 基线一：Naive RAG

### 原理

将语料库中每篇文章切分为句子，用 `sentence-transformers/all-MiniLM-L6-v2` 编码为向量，检索时计算余弦相似度返回 Top-K 句子。

### 安装依赖

```bash
pip install -r naive_rag/requirements.txt
```

### 步骤一：构建索引

```bash
# 从 corpus.json 构建（推荐）
python run_evaluation.py \
  --system naive_rag \
  --build_index \
  --corpus_json /Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json \
  --index_path ./naive_rag_index.pkl

# 或直接调用 indexer.py
python naive_rag/indexer.py \
  --corpus_json /Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json \
  --index_path ./naive_rag_index.pkl
```

### 步骤二：运行评估

```bash
python run_evaluation.py \
  --system naive_rag \
  --data_path /Users/ttung/Desktop/个人学习/data/multihop_rag \
  --index_path ./naive_rag_index.pkl \
  --top_k 5 \
  --num_samples 200 \
  --output_dir ./eval_results
```

### 快速测试检索

```bash
python naive_rag/retriever.py \
  --index_path ./naive_rag_index.pkl \
  --query "What is the relationship between OpenAI and Microsoft?" \
  --top_k 5
```

---

## 基线二：Microsoft GraphRAG 官方

### 前置条件

1. **Python 3.10+**（graphrag 要求）
2. **OpenAI API Key**（graphrag 需要调用 LLM 进行实体抽取和摘要生成）

### 安装

```bash
cd graphrag_official
bash setup.sh
```

### 配置 API Key

```bash
export GRAPHRAG_API_KEY="your-openai-api-key"
```

或编辑 `graphrag_official/graphrag_workspace/settings.yaml`，将 `api_key` 字段替换为实际值。

### 步骤一：构建索引

> ⚠️ **注意**：GraphRAG 索引构建需要大量 LLM API 调用，对于完整的 MultiHop-RAG 语料库（约 609 篇文章）可能需要数小时并消耗大量 API 配额。建议先用 `--max_docs 20` 测试。

```bash
# 快速测试（20 篇文档）
python graphrag_official/indexer.py \
  --corpus_json /Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json \
  --max_docs 20

# 完整索引（需要较长时间）
python graphrag_official/indexer.py \
  --corpus_json /Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json
```

### 步骤二：运行评估

```bash
# Local Search（微观检索）
python run_evaluation.py \
  --system graphrag_local \
  --data_path /Users/ttung/Desktop/个人学习/data/multihop_rag \
  --num_samples 50 \
  --output_dir ./eval_results

# Global Search（宏观检索）
python run_evaluation.py \
  --system graphrag_global \
  --data_path /Users/ttung/Desktop/个人学习/data/multihop_rag \
  --num_samples 50 \
  --output_dir ./eval_results
```

---

## 全系统对比评估

```bash
# 评估所有系统并生成对比报告
python run_evaluation.py \
  --system all \
  --data_path /Users/ttung/Desktop/个人学习/data/multihop_rag \
  --index_path ./naive_rag_index.pkl \
  --num_samples 100 \
  --output_dir ./eval_results
```

---

## 评估指标说明

| 指标 | 说明 | 适用系统 |
|------|------|---------|
| P@K | Precision at K，Top-K 中相关文档的比例 | Naive RAG, GraphRAG Local |
| R@K | Recall at K，Top-K 中召回的相关文档比例 | Naive RAG, GraphRAG Local |
| MRR | Mean Reciprocal Rank，第一个相关文档的排名倒数均值 | Naive RAG, GraphRAG Local |
| NDCG@K | Normalized DCG，考虑排名位置的检索质量 | Naive RAG, GraphRAG Local |
| ROUGE-L | 基于最长公共子序列的文本匹配分数 | 所有系统 |
| Avg Context Tokens | 平均 context token 数（效率指标） | 所有系统 |
| Avg Latency (ms) | 平均检索延迟 | 所有系统 |

---

## 结果输出

评估结果保存在 `--output_dir` 指定的目录下：

```
eval_results/
├── naive_rag_eval_results.json          # Naive RAG 详细结果
├── graphrag_local_eval_results.json     # GraphRAG Local 详细结果
├── graphrag_global_eval_results.json    # GraphRAG Global 详细结果
├── comparison_results.json              # 多系统对比 JSON
└── comparison_report.txt                # 多系统对比文本报告
```

---

## 常见问题

**Q: Naive RAG 索引构建很慢？**
A: 首次运行需要下载 all-MiniLM-L6-v2 模型（约 90MB）。可以增大 `--batch_size` 加速编码。

**Q: GraphRAG 报 API 错误？**
A: 确保设置了 `GRAPHRAG_API_KEY` 环境变量，且 API key 有效。

**Q: Python 版本不兼容？**
A: GraphRAG 需要 Python 3.10+。可以用 `pyenv install 3.10.14 && pyenv local 3.10.14` 切换版本。

**Q: 如何只评估部分数据？**
A: 使用 `--num_samples 50` 参数随机采样 50 条 QA 对进行快速验证。
