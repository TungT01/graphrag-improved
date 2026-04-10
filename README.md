# GraphRAG Improved

带**结构熵惩罚**的层次化 Leiden 聚类原型，对微软 [GraphRAG](https://github.com/microsoft/graphrag) 的社区检测模块进行改进。

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
│   ├── leiden_constrained.py  # 带结构熵惩罚的 Leiden 算法
│   ├── graphrag_workflow.py   # GraphRAG 兼容接口
│   └── tests/test_core.py     # 核心算法单元测试（25个）
│
├── data/ingestion.py          # 文档加载 + 分块（txt/json/pdf）
├── extraction/extractor.py    # 实体与关系抽取（rule/spacy）
├── output/reporter.py         # 结果输出（CSV / HTML 报告）
│
├── sample_data/               # 示例输入数据
└── tests/test_pipeline.py     # Pipeline 集成测试（24个）
```

## 快速开始

### 安装依赖

```bash
pip install networkx pandas pyyaml
# 可选：PDF 支持
pip install pypdf
# 可选：spaCy 后端
pip install spacy && python -m spacy download en_core_web_sm
# 可选：Parquet 导出
pip install pyarrow
```

### 运行示例

```bash
# 使用内置示例数据，一键运行
python -m graphrag_improved.main

# 指定自己的数据目录
python -m graphrag_improved.main --data-dir ./my_papers

# 调整物理约束强度
python -m graphrag_improved.main --lambda-init 500 --schedule cosine

# 仅执行数据摄入和抽取（调试模式）
python -m graphrag_improved.main --dry-run
```

### 输出文件

运行完成后，`output/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `report.html` | 可视化报告，含结构熵分布图 |
| `communities.csv` | 社区详情（含结构熵、λ 值） |
| `entities.csv` | 实体列表 |
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
```

## 测试

```bash
# 运行全部测试（49个）
python -m pytest graphrag_improved/constrained_leiden/tests/test_core.py \
                 graphrag_improved/tests/test_pipeline.py -v
```

测试覆盖：结构熵计算、退火机制、物理约束效果、层次输出、配置加载、数据摄入、实体抽取、端到端 Pipeline、输出文件完整性。

## 实验结果（示例数据）

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

- 命题转换（复合句拆解为原子命题）尚未实现
- spaCy 指代消解预处理尚未集成
- 需要在真实科研文献数据上做对照实验，验证检索质量提升效果
- 物理边界有效性依赖分块策略，段落分块效果最佳

## License

MIT
