# Changelog

本文件记录 GraphRAG Improved 项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased] — 架构重设计：物理优先 + 实例级节点

### 背景与动机

原版实现存在一个根本性的架构问题：实体在抽取阶段就被合并（实体消解），
所有文档里的"阿司匹林"变成同一个节点，结构熵约束只能在聚类阶段"软性"
阻止跨文档合并，物理信息（chunk_id）仅作为节点属性存在，不是图结构本身。

新架构的核心思想：**先物理、后语义。物理结构是一等公民，语义从物理结构中涌现。**

### 架构变更概览

#### 节点定义变更
- **旧**：实体节点，ID = `md5(title)`，携带 `primary_chunk_id` 属性
- **新**：实体实例节点，ID = `docX-paraY-sentZ-实体名`，携带完整物理路径

同一实体在不同句子里出现 = 不同节点，底层图保持物理纯净。

#### 边的定义变更
- **旧**：跨 chunk 的共现边，所有文档里共现过的实体之间都有边
- **新**：
  - 语义边：spaCy 依存句法提取的三元组 `(主语, 谓词, 宾语)`，主宾必须在同一句子内
  - 物理结构边：同句内的两个实体节点之间，权重 1.0
  - 跨句/跨段/跨文档：**无任何预设边**，连通性完全由聚类过程产生

底层图是由若干孤立的句子级子图组成的森林。

#### 实体消解变更
- **旧**：图构建阶段预处理，字符串匹配合并同名实体
- **新**：不在图构建阶段做，由 λ 退火驱动的 Leiden 聚类在社区层面涌现

#### 三元组提取变更
- **旧**：规则（正则匹配大写词）+ 共现窗口
- **新**：spaCy 依存句法分析，提取真正的主谓宾三元组
- 代词处理：主语或宾语为代词（PRP/PRP$）时直接跳过（方案一）
- 备选升级：若召回率不理想，切换至共指消解方案（coreferee/neuralcoref）

#### 句子切分变更
- **旧**：正则按标点切分
- **新**：spaCy `doc.sents`（依存句法树判断边界，处理缩写/引号/数字等边界歧义）

### Changed

#### `data/ingestion.py`
- 新增三级物理结构：`Document → Paragraph → Sentence`
- 新增 `SentenceUnit` 数据类，携带 `sent_id`（格式：`docX-paraY-sentZ`）
- `TextUnit` 升级为段落级容器，包含其下所有 `SentenceUnit`
- ID 生成规则：文档 ID 基于路径 hash，段落/句子 ID 基于序号

#### `extraction/extractor.py`
- 移除规则后端的共现窗口关系抽取
- 移除实体消解（`_disambiguate_entities`）
- 新增 spaCy 依存句法三元组提取后端
- 实体节点 ID 改为 `{sent_id}-{entity_name_normalized}`，包含完整物理路径
- `Entity.primary_chunk_id` 改为 `Entity.sent_id`（精确到句子级）

#### `constrained_leiden/graphrag_workflow.py`
- `build_graph_from_graphrag`：边构建逻辑重写，只在同句内建边
- `build_physical_nodes_from_graphrag`：物理锚点改为 `sent_id`（句子级）
- 移除 `min_edge_weight` 过滤（原用于过滤低频共现，新方案不适用）

#### `constrained_leiden/physical_anchor.py`
- `PhysicalNode.chunk_ids` 语义变更：由 chunk_id 集合改为 sent_id 集合
- 结构熵计算对象：社区内节点的句子 ID 分布熵（更细粒度的物理纯净度）

#### `retrieval/retriever.py`
- `BottomUpRetriever`：物理锚点检索改为 sent_id 级别定位
- 检索结果携带完整物理路径，支持精确溯源到原始句子

#### `experiments/run_experiment.py`
- `corpus_to_pipeline_text_units`：适配新的三级 ID 结构
- 评估时 ground-truth 对齐改为 doc_id 级别（MultiHop-RAG 的 evidence 是文章级）

---

## [v1.1.0] — 2026-04-14 — primary_chunk_id 修复 + 增量熵优化

### 背景
原版物理锚点使用 `text_unit_ids`（实体出现过的所有文档），导致高频实体
天然跨多个 chunk，结构熵无法降低，物理约束形同虚设。

### Changed
- `extraction/extractor.py`：新增 `primary_chunk_id` 字段，用 `Counter.most_common`
  确定主锚点（出现频次最高的 chunk）
- `constrained_leiden/graphrag_workflow.py`：`build_physical_nodes_from_graphrag`
  改为优先使用 `primary_chunk_id` 单一锚点
- `constrained_leiden/leiden_constrained.py`：新增 `CommunityEntropyState`，
  将 ΔH 计算从 O(|community|) 降到 O(1)，实测 6.8x 加速
- 移除 leidenalg 快速路径（事后拆分），统一使用纯 Python 过程约束实现

### Fixed
- 物理约束语义错误：100% 实体跨多 chunk → 修复后 Level-0 纯净率 100%

### Results
- 平均结构熵：0.8827 → 0.0000（-100%）
- Level-0 纯净率：24.73% → 100%
- MRR：+1.3%，P@5：+3.5%，NDCG@10：+1.9%

---

## [v1.0.0] — 初始版本

### Added
- 带结构熵惩罚的 Leiden 变体：J = Q_leiden - λ·H_structure
- λ 退火机制（指数/线性/余弦/阶梯）
- U-Retrieval 双轨检索（自顶向下社区导航 + 自底向上物理锚点）
- MultiHop-RAG 对照实验框架
- 评估指标：MRR / Precision@K / Recall@K / NDCG@K / 结构熵 / 纯净率
