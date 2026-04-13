# GraphRAG Improved

带**结构熵惩罚**的层次化 Leiden 聚类原型，对微软 [GraphRAG](https://github.com/microsoft/graphrag) 的社区检测、检索与评估模块进行全面改进。

## 核心思想

原版 GraphRAG 使用标准 Leiden 算法进行社区检测，纯粹以语义相似性为导向，可能将来自不同物理文档的实体错误地聚合在一起，导致社区摘要混杂多个文档的内容，降低检索精度。

本项目在目标函数中引入**结构熵惩罚项**，将聚类改造为三方权衡：

```
J = Q_leiden - λ · H_structure
```

| 符号 | 含义 |
|------|------|
| `Q_leiden` | 标准模块度增益（语义相似性） |
| `H_structure` | 社区内节点物理来源的香农熵（物理分散程度） |
| `λ` | 退火系数，随层级升高从极大值衰减至 0 |

**λ 退火机制**：底层 λ 极大，强制保持物理边界（同一文档的实体优先聚在一起）；高层 λ 趋零，释放跨文档语义融合能力，实现层次化的粒度控制。

在此基础上，本项目还新增了命题转换预处理、U-Retrieval 双轨检索和完整的评估框架，构成一个端到端的 GraphRAG 改进原型。

## 项目结构

```
graphrag_improved/
├── main.py                    # CLI 入口
├── run.py                     # Pipeline 编排（4步流程）
├── pipeline_config.py         # 强类型配置 + YAML 加载
├── config.yaml                # 项目配置文件
│
├── constrained_leiden/        # 核心算法
│   ├── physical_anchor.py     # 物理锚点 & 结构熵计算
│   ├── annealing.py           # λ 退火机制（4种曲线）
│   ├── leiden_constrained.py  # 带结构熵惩罚的 Leiden 算法（含细化阶段修复）
│   ├── graphrag_workflow.py   # GraphRAG 兼容接口
│   └── tests/test_core.py     # 核心算法单元测试（25个）
│
├── proposition/               # 命题转换预处理
│   └── transformer.py         # 共指消解 + 命题原子化
│
├── data/ingestion.py          # 文档加载 + 分块（txt/json/pdf）
├── extraction/extractor.py    # 实体与关系抽取（含实体消歧）
├── retrieval/retriever.py     # U-Retrieval 双轨检索
├── evaluation/evaluator.py    # 评估框架（Precision@K / MRR / NDCG / ROUGE-L）
├── output/reporter.py         # 结果输出（CSV / HTML 报告 + Force-directed 图）
│
├── sample_data/               # 示例输入数据
│   ├── sample_paper.txt
│   ├── sample_paper2.json
│   ├── sample_paper3.txt
│   └── sample_qa.json         # 示例 QA 对（用于检索评估）
└── tests/test_pipeline.py     # Pipeline 集成测试（24个）
```

## 新增模块说明

### 命题转换（`proposition/transformer.py`）

将原始文本块转换为原子命题，提升实体抽取和检索的精度。

**共指消解**：将代词和指代表达替换为其所指实体，使每个命题在脱离上下文后仍能独立理解。默认使用基于规则的轻量实现，可选升级到 spaCy + neuralcoref。

**命题原子化**：将复合句拆解为多个原子命题，每个命题只表达一个独立事实。支持并列连词分割、从句提取和括号内容独立化。

```python
from graphrag_improved.proposition.transformer import PropositionTransformer

transformer = PropositionTransformer(coref_backend="rule", atomize_backend="rule")
propositions = transformer.transform(text, chunk_id="doc1_chunk0")
# 返回 List[Proposition]，每个命题保留原始 chunk_id 作为物理锚点

# 批量处理并转回 TextUnit 格式
text_units = transformer.propositions_to_text_units(propositions)
```

### 实体消歧（`extraction/extractor.py`）

在原有规则/spaCy 双后端抽取的基础上，新增实体消歧步骤：将写法不同但指向同一实体的名称（如 `Graph RAG` 与 `GraphRAG`）自动合并，保留出现频率最高的写法作为规范名，并合并所有别名的 `chunk_ids`。

### U-Retrieval 双轨检索（`retrieval/retriever.py`）

实现论文中提出的 U-Retrieval 架构，结合两条互补的检索路径：

**自顶向下（Top-Down）**：从最高层社区出发，逐层向下导航，找到与查询最相关的社区，获取社区摘要作为全局上下文。

**自底向上（Bottom-Up）**：通过实体的物理锚点（`chunk_id`）直接定位原始文本块，提供精确的局部上下文。命中物理锚点的文本块额外获得 1.5× 权重加成。

两条路径的结果按可配置的 `alpha` 比例融合，提供兼顾全局语义和局部精确性的上下文，供下游 LLM 生成最终答案。默认使用 TF-IDF 相似度，可选升级到 BM25（`pip install rank-bm25`）。

```python
from graphrag_improved.retrieval.retriever import URetriever

retriever = URetriever.from_pipeline_result(
    pipeline_result, text_units,
    top_k_communities=5, top_k_chunks=5,
)
result = retriever.retrieve("What is the Leiden algorithm?")
print(result.merged_context)   # 融合后的上下文，直接传给 LLM
```

### 评估框架（`evaluation/evaluator.py`）

提供三类无需外部依赖的评估能力：

**社区质量评估**：模块度 Q、平均结构熵、Level 0 物理纯净率（熵 < 阈值的社区比例）、社区大小分布均匀性，以及各层结构熵的逐层统计。

**检索质量评估**：基于 QA 对和相关 `chunk_id` 标注，计算 Precision@K、Recall@K、F1@K（K 默认为 1/3/5/10）、MRR 和 NDCG@K。

**文本匹配评估**：用于 QA 生成质量评估，计算 Exact Match、Token-level F1 和 ROUGE-L（基于动态规划 LCS）。

```python
from graphrag_improved.evaluation.evaluator import Evaluator, load_qa_pairs_from_csv

evaluator = Evaluator()

# 社区质量（无需标注）
comm_metrics = evaluator.evaluate_community_quality(communities_df, relationships_df)
print(comm_metrics.summary())

# 检索质量（需要 QA 对）
qa_pairs = load_qa_pairs_from_csv("sample_data/sample_qa.json")
ret_metrics = evaluator.evaluate_retrieval(qa_pairs, retriever)
print(ret_metrics.summary())

# 一键生成完整报告
report = evaluator.full_report(communities_df, relationships_df, qa_pairs, retriever)
```

### HTML 报告增强（`output/reporter.py`）

在原有结构熵分布折线图的基础上，新增基于 Canvas 的 **Force-directed 知识图谱可视化**：节点颜色代表所属社区（Level 0 着色），边粗细代表共现权重，支持鼠标拖拽交互。图谱最多展示 80 个节点，自动过滤孤立边，120 帧物理模拟后静止。

### Leiden 细化阶段修复（`constrained_leiden/leiden_constrained.py`）

修复了原版 Leiden 算法细化（refinement）阶段在处理小图时可能出现的边界条件错误，确保在节点数极少（< 3）或社区只有单节点时算法仍能正确收敛，不抛出异常。

## 快速开始

### 安装依赖

```bash
pip install networkx pandas pyyaml

# 可选：PDF 支持
pip install pypdf

# 可选：spaCy 后端（实体抽取 / 共指消解）
pip install spacy && python -m spacy download en_core_web_sm

# 可选：BM25 检索后端
pip install rank-bm25

# 可选：Parquet 导出
pip install pyarrow
```

### 运行示例

```bash
# 使用内置示例数据，一键运行
python -m graphrag_improved.main

# 指定自己的数据目录
python -m graphrag_improved.main --data-dir ./my_papers

# 调整物理约束强度和退火曲线
python -m graphrag_improved.main --lambda-init 500 --schedule cosine

# 仅执行数据摄入和抽取（调试模式）
python -m graphrag_improved.main --dry-run
```

### 使用 U-Retrieval 检索

```python
from graphrag_improved.run import run_pipeline
from graphrag_improved.pipeline_config import PipelineConfig
from graphrag_improved.retrieval.retriever import URetriever

config = PipelineConfig.from_yaml("config.yaml")
result, text_units = run_pipeline(config)

retriever = URetriever.from_pipeline_result(result, text_units)
ret_result = retriever.retrieve(
    "What is structural entropy?",
    entity_mentions=["Leiden", "GraphRAG"],
    alpha=0.5,   # 0.5 = 社区上下文与原文片段各占一半
)
print(ret_result.merged_context)
```

### 使用命题转换预处理

```python
from graphrag_improved.proposition.transformer import PropositionTransformer

transformer = PropositionTransformer()
# 将文本块列表转换为原子命题，再送入抽取模块
prop_units = transformer.transform_batch(text_units)
text_units_for_extraction = transformer.propositions_to_text_units(prop_units)
```

### 输出文件

运行完成后，`output/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `report.html` | 可视化报告，含结构熵分布图 + Force-directed 知识图谱 |
| `communities.csv` | 社区详情（含结构熵、λ 值） |
| `entities.csv` | 实体列表（含消歧后的规范名） |
| `relationships.csv` | 关系列表 |
| `summary.txt` | 控制台摘要文本 |

## 配置说明

编辑 `config.yaml` 调整所有参数，或通过 CLI 参数临时覆盖：

```yaml
clustering:
  lambda_init: 1000.0        # 底层物理约束强度
  lambda_min: 0.0            # 高层约束下限
  annealing_schedule: "exponential"  # exponential | linear | cosine | step
  decay_rate: 0.5
  max_cluster_size: 10

extraction:
  backend: "rule"            # rule | spacy
  min_entity_freq: 1
  cooccurrence_window: 3

retrieval:
  top_k_communities: 5
  top_k_chunks: 5
  max_context_chars: 4000
```

## 测试

```bash
# 运行全部测试（48 passed, 1 skipped）
python -m pytest graphrag_improved/constrained_leiden/tests/test_core.py \
                 graphrag_improved/tests/test_pipeline.py -v
```

测试覆盖：结构熵计算、退火机制、物理约束效果、层次输出、Leiden 细化阶段边界条件、配置加载、数据摄入、实体抽取与消歧、命题转换、U-Retrieval 双轨检索、评估框架（Precision@K / MRR / NDCG / ROUGE-L）、端到端 Pipeline、输出文件完整性。

## 实验结果

### MultiHop-RAG 真实数据集对照实验

在 [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) 数据集上进行了完整的对照实验：

**数据规模**：609 篇新闻文章，2255 条有效 QA 对（含 inference / comparison / temporal / null 四种问题类型）

**实验设置**：
- Baseline：原版 Leiden（λ=0，无结构熵约束）
- Ours：结构熵约束 Leiden（λ=1000，指数退火）
- 检索方式：U-Retrieval 双轨检索（TF-IDF + 社区实体提及加权）
- 评估指标：Precision@K / Recall@K / MRR / NDCG@K

**检索质量对比**（修复评估偏差后的公平结果）：

| 指标 | Baseline (λ=0) | Ours (λ=1000) | 差异 |
|------|---------------|---------------|------|
| MRR | 0.4066 | 0.4070 | +0.1% |
| Precision@1 | 0.3273 | 0.3273 | 0.0% |
| Precision@5 | 0.1435 | 0.1439 | +0.3% |
| Recall@5 | 0.2927 | 0.2935 | +0.3% |
| Recall@10 | 0.3806 | 0.3811 | +0.1% |
| NDCG@10 | 0.3106 | 0.3110 | +0.1% |

**社区质量对比**：

| 指标 | Baseline (λ=0) | Ours (λ=1000) | 变化 |
|------|---------------|---------------|------|
| 社区数量 | 200 | 907 | +4.5x |
| 平均结构熵 | 2.8483 | **2.2151** | **-22.2%** |

**结论**：在控制变量公平的实验设计下（bottom-up 检索独立于社区划分、top_down_ids 总量对等、F1 逐查询计算），两个版本的检索指标差异极小（<0.3%），统计上不显著。但结构熵约束 Leiden 将社区平均结构熵降低了 **22.2%**，产生了更细粒度、物理边界更集中的社区划分（907 个 vs 200 个），这是算法的直接效果。

**说明**：当前 TF-IDF 检索框架下，社区划分质量对检索指标的影响有限，因为 bottom-up 的 TF-IDF 检索本身已经足够精准。结构熵约束的优势在接入向量检索（FAISS/Chroma）或 LLM 生成摘要后，通过更精准的社区摘要质量体现，是后续工作的重点方向。

**运行实验**：

```bash
# 下载数据集
git clone https://github.com/yixuantt/MultiHop-RAG
mkdir -p data/multihop_rag
cp MultiHop-RAG/dataset/corpus.json data/multihop_rag/
cp MultiHop-RAG/dataset/MultiHopRAG.json data/multihop_rag/

# 安装加速依赖
pip install leidenalg igraph

# 运行对照实验（全量 2556 条 QA，约 100 秒）
python -m graphrag_improved.experiments.run_experiment \
    --data-dir ./data/multihop_rag \
    --output-dir ./experiments/results
```

### 示例数据快速验证

以内置示例数据运行，耗时 0.02 秒：

- 24 个文本块 → 44 个实体 → 68 条关系 → **53 个社区，8 个层次**
- Level 0 平均结构熵 **0.300**，Level 7 平均结构熵 **1.731**，随层级单调递增，符合 λ 退火预期
- Level 0 物理纯净社区（熵 < 0.01）占比 **64.3%**

## 与原版 GraphRAG 的关系

本项目的 `graphrag_workflow.py` 提供了与原版 GraphRAG 兼容的接口，可直接替换其 `create_communities` 工作流：

```python
from graphrag_improved.constrained_leiden.graphrag_workflow import run_constrained_community_detection

# 输入：GraphRAG 标准的 entities / relationships DataFrame
# 输出：GraphRAG 兼容的 communities DataFrame（额外含 structural_entropy 列）
communities_df = run_constrained_community_detection(entities_df, relationships_df)
```

## 局限性与后续方向

- 命题转换目前使用规则后端，精度有限；接入 LLM 后端可显著提升原子化质量
- 共指消解规则策略仅处理简单代词，复杂指代链需要 neuralcoref / fastcoref
- 物理边界有效性依赖分块策略，段落分块效果最佳
- U-Retrieval 当前使用 TF-IDF，接入向量检索（FAISS / Chroma）可进一步提升召回率
- 当前实验在新闻领域数据集上验证，后续可扩展到科研文献（如 HotpotQA、MuSiQue）
- 结构熵约束的最优 λ 值依赖数据集特性，可通过网格搜索自动调优

## License

MIT
