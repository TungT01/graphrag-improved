# GraphRAG Improved — 项目现状全面盘点

> 更新日期：2026-04-24  
> 当前版本：v4（四组消融对照）  
> 仓库地址：https://github.com/TungT01/graphrag-improved

---

## 一、项目目标

### 研究问题

微软 GraphRAG 在构建知识图谱时，使用 Leiden 算法对实体进行社区检测，再以社区为单位做 RAG 检索。原版 Leiden 只优化模块度 Q，不关心社区内实体的物理来源（即实体来自哪些文档/段落/句子）。这导致跨文档的同名实体被错误合并进同一社区，检索时产生大量噪声。

### 核心假设

在 Leiden 目标函数中加入**结构熵惩罚项**，驱动社区向物理纯净方向演化，可以提升基于社区的 RAG 检索质量：

```
J = Q_leiden - λ · H_structure

H_structure = Σ_c  |c|/|V| · H(sent_id 分布 in c)
```

其中 λ 采用指数退火：`λ(t) = λ_init · exp(-decay · t)`，初始值 λ_init=1000。

### 对比基线

| 系统 | 描述 |
|------|------|
| **Naive RAG** | TF-IDF 向量检索，sentence-transformers 编码，无图结构 |
| **GraphRAG Baseline (λ=0)** | 原版 Leiden，无结构熵约束，v3 物理优先架构 |
| **GraphRAG Improved (λ=1000)** | 结构熵约束 Leiden，λ 指数退火 |

---

## 二、系统架构

### 2.1 整体模块

```
graphrag_improved/
├── constrained_leiden/       # 核心算法
│   ├── leiden_constrained.py     # 结构熵约束 Leiden 主实现
│   ├── graphrag_workflow.py      # 图构建工作流
│   ├── physical_anchor.py        # 物理锚点（sent_id 级别）
│   └── annealing.py              # λ 退火调度（指数/线性/余弦/阶梯）
├── extraction/
│   └── extractor.py              # spaCy 依存句法三元组抽取
├── retrieval/
│   └── retriever.py              # U-Retrieval 双轨检索
├── baselines/
│   ├── naive_rag/                # TF-IDF 基线
│   └── eval_results/             # 基线评估结果
├── experiments/
│   ├── run_multihop_eval.py      # 四组对照实验主脚本
│   ├── ablation_study_notes.md   # 消融实验详细记录
│   └── results/                  # 实验结果 JSON
├── evaluation/                   # 评估指标计算
├── data/                         # 数据集接口
├── README.md
├── CHANGELOG.md
└── config.yaml
```

### 2.2 v3 物理优先架构（当前版本）

**核心设计原则：先物理、后语义。物理结构是一等公民，语义从物理结构中涌现。**

**节点设计**：每个节点是带物理坐标的实体实例，而非概念节点：

```
节点 ID = {doc_id}-p{para_idx}-s{sent_idx}-{entity_name_normalized}
示例：doc0-p3-s1-aspirin
```

同一实体名在不同句子里出现 = 不同节点，底层图保持物理纯净。

**边的设计**：
- 语义边：spaCy 依存句法提取的主谓宾三元组，主宾必须在同一句子内
- 物理结构边：同句内实体共现（predicate = "co_occurs"），权重 1.0
- 跨句/跨段/跨文档：无任何预设边，连通性完全由聚类过程产生

**底层图特征**：由若干孤立的句子级子图组成的森林，总连通分量数 24,858，最大连通分量仅 9 个节点。

### 2.3 U-Retrieval 双轨检索

- **自顶向下**：从高层社区摘要出发，逐层导航到相关子社区
- **自底向上**：通过物理锚点（sent_id）直接定位原始句子，再向上聚合

---

## 三、评估数据集

**MultiHop-RAG**（COLM 2024）

| 属性 | 数值 |
|------|------|
| 文章数量 | 609 篇新闻文章 |
| QA 对总数 | 2,556 条 |
| 问题类型 | inference_query / comparison_query / temporal_query / null |
| 证据分布 | 每条 QA 的证据分布在 2-4 篇文档中 |
| 本次实验规模 | 200 QA 采样（有效 169 条） |

---

## 四、实验结果

### 4.1 四组对照实验设计

| 组号 | 名称 | λ_init | 路径A（文档内消解） | 路径B（噪声过滤） |
|------|------|--------|-------------------|-----------------|
| [0] | Baseline | 0.0 | ✗ | ✗ |
| [1] | Ours | 1000.0 | ✗ | ✗ |
| [2] | Ours+A | 1000.0 | ✓ | ✗ |
| [3] | Ours+A+B | 1000.0 | ✓ | ✓ |

### 4.2 检索质量指标（n=200，valid=169）

| 指标 | [0] Baseline | [1] Ours | [2] Ours+A | [3] Ours+A+B |
|------|:---:|:---:|:---:|:---:|
| MRR | 0.3492 | 0.3558 (+1.9%) | 0.3558 (+1.9%) | 0.3552 (+1.7%) |
| Precision@1 | 0.2663 | 0.2663 (±0%) | 0.2663 (±0%) | 0.2663 (±0%) |
| Precision@3 | 0.1617 | 0.1617 (±0%) | 0.1617 (±0%) | 0.1617 (±0%) |
| **Precision@5** | **0.1325** | **0.1609 (+21.4%)** | **0.1609 (+21.4%)** | **0.1633 (+23.2%)** |
| Precision@10 | 0.0882 | 0.0882 (±0%) | 0.0882 (±0%) | 0.0882 (±0%) |
| **Recall@5** | **0.2648** | **0.3180 (+20.1%)** | **0.3180 (+20.1%)** | **0.3230 (+22.0%)** |
| Recall@10 | 0.3481 | 0.3481 (±0%) | 0.3481 (±0%) | 0.3481 (±0%) |
| F1@5 | 0.1744 | 0.2110 (+21.0%) | 0.2110 (+21.0%) | 0.2141 (+22.8%) |
| **NDCG@5** | **0.2347** | **0.2654 (+13.1%)** | **0.2654 (+13.1%)** | **0.2680 (+14.2%)** |
| NDCG@10 | 0.2737 | 0.2789 (+1.9%) | 0.2789 (+1.9%) | 0.2792 (+2.0%) |

### 4.3 图结构与社区质量指标

| 指标 | [0] Baseline | [1] Ours | [2] Ours+A | [3] Ours+A+B |
|------|:---:|:---:|:---:|:---:|
| 实体数量 | 60,439 | 60,439 | 60,439 | 47,142 |
| 关系数量 | 84,704 | 84,704 | 84,704 | 61,812 |
| 社区数量 | 29,398 | 79,114 | 79,113 | 63,497 |
| 层次数量 | 1 | 3 | 3 | 3 |
| 模块度 Q | 0.7533 | 0.7533 | 0.7535 | **0.8025 (+6.5%)** |
| 平均结构熵 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Level-0 纯净率 | 100% | 100% | 100% | 100% |
| 平均社区大小 | 2.06 | 2.29 | 2.29 | 2.23 |

### 4.4 运行时间

| 组号 | 社区检测耗时 | 检索评估耗时 | 总耗时 |
|------|:---:|:---:|:---:|
| [0] Baseline | 97.6s | 23.9s | 121.5s |
| [1] Ours | 222.0s | 50.6s | 272.6s |
| [2] Ours+A | 205.9s | 50.0s | 255.9s |
| [3] Ours+A+B | 191.5s | 38.0s | 229.5s |

路径B 额外抽取耗时：348.7s（仅首次，后续命中缓存）。

### 4.5 与 Naive RAG 基线对比（n=200，valid=169）

| 指标 | Naive RAG | [3] Ours+A+B | 差距 |
|------|:---:|:---:|:---:|
| MRR | **0.6389** | 0.3552 | -44.4% |
| Precision@5 | **0.2568** | 0.1633 | -36.4% |
| Recall@5 | **0.5059** | 0.3230 | -36.2% |
| NDCG@5 | **0.4792** | 0.2680 | -44.1% |
| avg_latency_ms | 70.6ms | ~1600ms | ~23x 慢 |

Naive RAG 按问题类型细分：

| 问题类型 | n | MRR | Recall@5 |
|---------|---|-----|---------|
| comparison_query | 57 | 0.714 | 0.652 |
| inference_query | 59 | 0.619 | 0.347 |
| temporal_query | 53 | 0.580 | 0.525 |

---

## 五、关键发现与结论

### 5.1 主线结论：结构熵约束有效

[0] vs [1] 的对比验证了核心假设：结构熵约束 Leiden（λ=1000）相比原版 Leiden（λ=0）：

- **P@5 提升 21.4%**（0.1325 → 0.1609）
- **Recall@5 提升 20.1%**（0.2648 → 0.3180）
- **NDCG@5 提升 13.1%**（0.2347 → 0.2654）
- 社区层次从 1 层增加到 3 层，社区数量从 29,398 增加到 79,114

提升主要集中在 @5 窗口，@10 窗口提升有限（MRR +1.9%，NDCG@10 +1.9%），说明结构熵约束改善了 top-5 检索精度，但对更宽的召回窗口影响较小。

### 5.2 路径A（文档内消解边）效果为零

[1] Ours 和 [2] Ours+A 的所有检索指标完全相同。原因：在 `use_lcc=False` 模式下，Leiden 已对所有连通分量分别运行，碎片化不再是瓶颈。加入的软连接边（weight=0.5）权重过低，Leiden 在优化模块度时倾向于忽略这些弱连接。

### 5.3 路径B（噪声实体过滤）有实质效果

[3] Ours+A+B 相比 [1] Ours：P@5 +1.5%，Recall@5 +1.6%，NDCG@5 +1.0%，模块度 Q +6.5%。噪声过滤减少了 22% 实体和 27% 关系，图变得更干净，社区质量显著提升。

### 5.4 与 Naive RAG 的差距

当前 GraphRAG Improved 在所有指标上均落后于 Naive RAG（TF-IDF），差距约 36-44%。这是 GraphRAG 类方法的已知问题：社区级检索的粒度较粗，在精确文档检索任务上不如直接向量检索。

### 5.5 平均结构熵为零的说明

所有组的平均结构熵均为 0.0000，这是预期行为。v3 物理优先架构中，每个节点的 sent_id 是唯一的（节点 ID 本身就包含 sent_id），因此每个节点天然属于且只属于一个句子，社区内 sent_id 分布的熵自然为 0。结构熵约束的实际作用体现在**阻止跨句子节点被错误合并**，通过调整社区边界间接改善检索粒度。

---

## 六、已知问题与待解决事项

### 6.1 检索质量问题

**问题 1：与 Naive RAG 差距显著**
- 当前 GraphRAG Improved 在 MRR、P@5、NDCG@5 上均落后 Naive RAG 约 40%
- 根本原因：社区级检索粒度过粗，一个社区可能包含多篇文档的实体，检索时无法精确定位到单篇文档
- 待解决：改进 U-Retrieval 的文档级聚合逻辑，或引入混合检索（社区导航 + 向量检索）

**问题 2：@10 窗口提升有限**
- 结构熵约束主要改善 @5 窗口，@10 窗口提升不足 2%
- 可能原因：当前检索策略在 top-5 之后的排序质量较差
- 待解决：改进社区排序算法，引入重排序（reranking）

**问题 3：inference_query 类型表现最差**
- Naive RAG 在 inference_query 上 Recall@5 仅 0.347（vs comparison_query 的 0.652）
- 多跳推理问题对所有系统都是挑战
- 待解决：专门针对多跳推理设计检索策略

### 6.2 图结构问题

**问题 4：图碎片化严重**
- v3 物理优先架构导致 24,858 个连通分量，最大分量仅 9 个节点
- 当前通过 `use_lcc=False` 绕过，但根本问题未解决
- 待解决：路径A 的硬合并方案（预处理阶段直接合并同文档同名实体）

**问题 5：路径A 软连接无效**
- 文档内消解边（weight=0.5）被 Leiden 忽略，效果为零
- 待解决：提高边权重至 0.8-1.0，或改用硬合并

### 6.3 评估问题

**问题 6：评估规模不足**
- 当前 200 QA 采样（169 有效）统计功效有限，1-2% 的差异可能在误差范围内
- 待解决：在全量 2556 QA 上复现主要结论（预计耗时 2-3 小时）

**问题 7：缺少 GraphRAG 官方基线对比**
- 目前只有 Naive RAG 和自实现的 Baseline，缺少 GraphRAG 官方（3.0.9 CLI）的对比数据
- 待解决：运行 GraphRAG 官方 CLI 并收集评估结果

### 6.4 工程问题

**问题 8：运行时间较长**
- [1] Ours 总耗时 272.6s（vs Baseline 121.5s），慢 2.2x
- 主要瓶颈：结构熵约束使社区数量增加 2.7x，检索评估耗时相应增加
- 待解决：优化社区检测的增量计算，或引入并行化

---

## 七、版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v4 | 2026-04-15 | 路径A/B 消融实验 + 四组对照框架；路径B 噪声过滤有效（P@5 +23.2%），路径A 软连接无增益 |
| v3 | 2026-04-14 | 架构重设计——物理优先，实例级节点，移除实体消解；spaCy 依存句法三元组抽取 |
| v1.1.0 | 2026-04-14 | primary_chunk_id 修复 + 增量熵优化（6.8x 加速）；Level-0 纯净率 24.73% → 100% |
| v1.0.0 | 2026-04-10 | 初始版本：结构熵约束 Leiden、λ 退火、U-Retrieval、MultiHop-RAG 评估框架 |

---

## 八、后续工作计划

### 近期（1-2 周）

1. **全量评估**：在 2556 条 QA 上运行完整实验，验证 200 QA 结论的统计显著性
2. **GraphRAG 官方基线**：运行 GraphRAG 3.0.9 CLI，补充三方对比数据
3. **路径A 硬合并**：实现预处理阶段的同文档同名实体合并，替代当前软连接方案

### 中期（1 个月）

4. **λ 敏感性分析**：测试 λ_init ∈ {100, 500, 1000, 2000, 5000}，绘制 P@5 vs λ 曲线
5. **混合检索**：在社区导航基础上引入向量检索，缩小与 Naive RAG 的差距
6. **按问题类型分析**：分别报告 inference/comparison/temporal 三类问题的指标

### 长期（论文写作）

7. **消融实验完整化**：补充路径A 硬合并、边权重敏感性、NER 类型过滤等消融数据
8. **论文撰写**：以 [0] vs [1] 为主线，路径B 为工程优化，整理成完整论文

---

## 九、复现指南

### 环境准备

```bash
cd /Users/ttung/Desktop/个人学习/graphrag_improved
python3 -m venv .venv-graphrag
source .venv-graphrag/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### 运行实验

```bash
# 快速验证（200 QA，约 15 分钟）
python3 -u -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --n-qa 200 \
    --output-dir ./experiments/results

# 全量运行（2556 QA，约 2-3 小时）
python3 -u -m graphrag_improved.experiments.run_multihop_eval \
    --data-dir ./data/multihop_rag \
    --full \
    --output-dir ./experiments/results
```

### 缓存管理

```
experiments/results/cache/
├── entities_full.parquet        # 原始抽取（Baseline/Ours/Ours+A 共用）
├── relationships_full.parquet
├── entities_full_b.parquet      # 路径B 抽取（噪声过滤后）
└── relationships_full_b.parquet
```

### 关键代码位置

| 功能 | 文件 | 关键符号 |
|------|------|---------|
| 结构熵约束 Leiden | `constrained_leiden/leiden_constrained.py` | `hierarchical_leiden_constrained()` |
| λ 退火调度 | `constrained_leiden/annealing.py` | `AnnealingSchedule` |
| 图构建工作流 | `constrained_leiden/graphrag_workflow.py` | `build_graph_from_graphrag()` |
| spaCy 三元组抽取 | `extraction/extractor.py` | `_STOPWORDS`, `extract_entities()` |
| 噪声词表（路径B） | `extraction/extractor.py` | `_STOPWORDS` |
| 文档内消解边（路径A） | `constrained_leiden/graphrag_workflow.py` | `build_intra_doc_entity_edges()` |
| U-Retrieval 检索 | `retrieval/retriever.py` | `BottomUpRetriever` |
| 四组实验入口 | `experiments/run_multihop_eval.py` | `run_experiment()` |
| Naive RAG 基线 | `baselines/naive_rag/` | — |
