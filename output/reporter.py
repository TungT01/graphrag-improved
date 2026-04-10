"""
output/reporter.py
------------------
结果输出模块：将 Pipeline 运行结果持久化并生成可读报告。

输出内容：
  1. communities.parquet  — GraphRAG 兼容格式，供下游使用
  2. communities.csv      — 人类可读的社区摘要表
  3. entities.csv         — 实体列表
  4. relationships.csv    — 关系列表
  5. report.html          — 可视化 HTML 报告（含社区结构、熵分布图表）
  6. summary.txt          — 控制台摘要文本
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..pipeline_config import OutputConfig


# ---------------------------------------------------------------------------
# Pipeline 运行结果容器
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    封装整个 Pipeline 的运行结果，作为 reporter 的唯一输入。

    Attributes
    ----------
    communities_df   : 社区检测结果（含 structural_entropy, lambda_used）
    entities_df      : 实体表
    relationships_df : 关系表
    run_stats        : 运行统计信息（耗时、节点数等）
    config_snapshot  : 本次运行的配置快照（用于报告存档）
    """
    communities_df: pd.DataFrame
    entities_df: pd.DataFrame
    relationships_df: pd.DataFrame
    run_stats: Dict = None
    config_snapshot: Dict = None

    def __post_init__(self):
        if self.run_stats is None:
            self.run_stats = {}
        if self.config_snapshot is None:
            self.config_snapshot = {}


# ---------------------------------------------------------------------------
# 控制台摘要
# ---------------------------------------------------------------------------

def print_console_summary(result: PipelineResult) -> str:
    """
    打印并返回控制台摘要文本。

    Returns
    -------
    str
        摘要文本（同时打印到 stdout）
    """
    df = result.communities_df
    entities_df = result.entities_df
    rels_df = result.relationships_df
    stats = result.run_stats

    lines = []
    lines.append("=" * 60)
    lines.append("  GraphRAG Improved — 运行结果摘要")
    lines.append("=" * 60)

    # 基础统计
    lines.append(f"\n📄 输入统计")
    lines.append(f"   文本块数量   : {stats.get('num_text_units', 'N/A')}")
    lines.append(f"   实体数量     : {len(entities_df)}")
    lines.append(f"   关系数量     : {len(rels_df)}")

    # 社区统计
    if not df.empty:
        num_levels = df["level"].nunique()
        lines.append(f"\n🔗 社区结构")
        lines.append(f"   层次数量     : {num_levels}")
        lines.append(f"   社区总数     : {len(df)}")

        for level in sorted(df["level"].unique()):
            level_df = df[df["level"] == level]
            avg_entropy = level_df["structural_entropy"].mean()
            avg_size = level_df["entity_ids"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            ).mean()
            lines.append(
                f"   Level {level:<2}     : {len(level_df):>4} 个社区 | "
                f"平均大小 {avg_size:.1f} | 平均结构熵 {avg_entropy:.4f}"
            )

        # 物理约束效果
        level0_df = df[df["level"] == 0]
        if not level0_df.empty:
            pure_communities = (level0_df["structural_entropy"] < 0.01).sum()
            purity_rate = pure_communities / len(level0_df) * 100
            lines.append(f"\n🔒 物理约束效果 (Level 0)")
            lines.append(f"   物理纯净社区 : {pure_communities}/{len(level0_df)} ({purity_rate:.1f}%)")
            lines.append(f"   平均结构熵   : {level0_df['structural_entropy'].mean():.4f}")
            lines.append(f"   最大结构熵   : {level0_df['structural_entropy'].max():.4f}")

    # 耗时
    if "elapsed_seconds" in stats:
        lines.append(f"\n⏱  运行耗时     : {stats['elapsed_seconds']:.2f} 秒")

    lines.append("\n" + "=" * 60)
    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# CSV 导出
# ---------------------------------------------------------------------------

def export_csv(result: PipelineResult, output_dir: str) -> Dict[str, str]:
    """
    导出 CSV 文件，返回 {文件名: 路径} 映射。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported = {}

    # 社区表（展开 list 列为字符串，便于 Excel 打开）
    if not result.communities_df.empty:
        comm_csv = result.communities_df.copy()
        for col in ["entity_ids", "relationship_ids", "text_unit_ids", "children"]:
            if col in comm_csv.columns:
                comm_csv[col] = comm_csv[col].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else str(x or "")
                )
        path = str(out / "communities.csv")
        comm_csv.to_csv(path, index=False, encoding="utf-8-sig")
        exported["communities.csv"] = path

    # 实体表
    if not result.entities_df.empty:
        ent_csv = result.entities_df.copy()
        if "text_unit_ids" in ent_csv.columns:
            ent_csv["text_unit_ids"] = ent_csv["text_unit_ids"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else str(x or "")
            )
        path = str(out / "entities.csv")
        ent_csv.to_csv(path, index=False, encoding="utf-8-sig")
        exported["entities.csv"] = path

    # 关系表
    if not result.relationships_df.empty:
        path = str(out / "relationships.csv")
        result.relationships_df.to_csv(path, index=False, encoding="utf-8-sig")
        exported["relationships.csv"] = path

    return exported


# ---------------------------------------------------------------------------
# Parquet 导出
# ---------------------------------------------------------------------------

def export_parquet(result: PipelineResult, output_dir: str) -> Dict[str, str]:
    """导出 Parquet 文件（GraphRAG 兼容格式）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported = {}

    for name, df in [
        ("communities", result.communities_df),
        ("entities", result.entities_df),
        ("relationships", result.relationships_df),
    ]:
        if not df.empty:
            path = str(out / f"{name}.parquet")
            df.to_parquet(path, index=False)
            exported[f"{name}.parquet"] = path

    return exported


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------

def _build_entropy_chart_data(communities_df: pd.DataFrame) -> str:
    """构建结构熵分布图的 JSON 数据。"""
    if communities_df.empty:
        return "[]"

    chart_data = []
    for level in sorted(communities_df["level"].unique()):
        level_df = communities_df[communities_df["level"] == level]
        entropies = level_df["structural_entropy"].tolist()
        chart_data.append({
            "level": int(level),
            "entropies": [round(e, 4) for e in entropies],
            "mean": round(float(level_df["structural_entropy"].mean()), 4),
            "lambda": round(float(level_df["lambda_used"].iloc[0]), 4),
        })
    return json.dumps(chart_data)


def _build_community_table_html(communities_df: pd.DataFrame) -> str:
    """构建社区详情 HTML 表格。"""
    if communities_df.empty:
        return "<p>暂无社区数据</p>"

    rows = []
    display_df = communities_df.sort_values(["level", "community_id"]).head(200)

    for _, row in display_df.iterrows():
        entity_ids = row.get("entity_ids", [])
        entity_count = len(entity_ids) if isinstance(entity_ids, list) else 0
        entropy = row.get("structural_entropy", 0.0)
        entropy_class = "entropy-low" if entropy < 0.1 else (
            "entropy-mid" if entropy < 0.5 else "entropy-high"
        )
        rows.append(f"""
        <tr>
            <td>{row.get('level', '')}</td>
            <td>{row.get('community_id', '')}</td>
            <td>{entity_count}</td>
            <td class="{entropy_class}">{entropy:.4f}</td>
            <td>{row.get('lambda_used', 0.0):.2f}</td>
            <td class="title-cell">{row.get('title', '')}</td>
        </tr>""")

    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>层级</th><th>社区 ID</th><th>实体数</th>
                <th>结构熵</th><th>λ 值</th><th>标题</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    {"<p class='note'>（仅显示前 200 条）</p>" if len(communities_df) > 200 else ""}
    """


def _build_graph_data(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    max_nodes: int = 80,
) -> str:
    """
    构建 Force-directed 图可视化所需的 JSON 数据。

    只取 Level 0 社区的实体和关系，节点按社区着色。
    为避免图过于密集，最多取 max_nodes 个节点。
    """
    if entities_df.empty or relationships_df.empty:
        return json.dumps({"nodes": [], "links": []})

    # 构建实体 → 社区 ID 的映射（Level 0）
    entity_to_comm: dict = {}
    if not communities_df.empty:
        level0 = communities_df[communities_df["level"] == 0]
        for _, row in level0.iterrows():
            comm_id = int(row.get("community_id", 0))
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, list):
                for eid in entity_ids:
                    entity_to_comm[str(eid)] = comm_id

    # 取前 max_nodes 个实体
    display_entities = entities_df.head(max_nodes)
    entity_set = set(display_entities["title"].astype(str).tolist())

    # 构建节点列表
    nodes = []
    for _, row in display_entities.iterrows():
        title = str(row.get("title", ""))
        eid = str(row.get("id", title))
        comm_id = entity_to_comm.get(eid, entity_to_comm.get(title, -1))
        nodes.append({
            "id": title,
            "type": str(row.get("type", "UNKNOWN")),
            "community": comm_id,
            "entropy": 0.0,
        })

    # 构建边列表（只保留两端都在节点集合中的边）
    links = []
    seen_pairs: set = set()
    for _, row in relationships_df.iterrows():
        src = str(row.get("source", ""))
        tgt = str(row.get("target", ""))
        if src not in entity_set or tgt not in entity_set:
            continue
        pair = (min(src, tgt), max(src, tgt))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        links.append({
            "source": src,
            "target": tgt,
            "weight": float(row.get("weight", 1.0)),
        })

    return json.dumps({"nodes": nodes, "links": links})


def export_html_report(result: PipelineResult, output_dir: str) -> str:
    """
    生成 HTML 可视化报告。

    Returns
    -------
    str
        HTML 文件路径
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = result.communities_df
    stats = result.run_stats
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统计数据
    num_levels = df["level"].nunique() if not df.empty else 0
    num_communities = len(df)
    num_entities = len(result.entities_df)
    num_relations = len(result.relationships_df)
    level0_entropy = (
        df[df["level"] == 0]["structural_entropy"].mean()
        if not df.empty and 0 in df["level"].values else 0.0
    )

    entropy_chart_data = _build_entropy_chart_data(df)
    community_table = _build_community_table_html(df)
    graph_data = _build_graph_data(result.entities_df, result.relationships_df, df)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphRAG Improved — 运行报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f7fa; color: #333; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             color: white; padding: 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  .header p {{ opacity: 0.75; font-size: 14px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 16px; margin-bottom: 32px; }}
  .stat-card {{ background: white; border-radius: 12px; padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .value {{ font-size: 36px; font-weight: 700; color: #0f3460; }}
  .stat-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
  .section {{ background: white; border-radius: 12px; padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
  .section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 16px;
                 padding-bottom: 12px; border-bottom: 2px solid #f0f0f0; }}
  .chart-container {{ position: relative; height: 300px; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ background: #f8f9fa; padding: 10px 12px; text-align: left;
                    font-weight: 600; border-bottom: 2px solid #e9ecef; }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
  .data-table tr:hover td {{ background: #f8f9fa; }}
  .entropy-low {{ color: #28a745; font-weight: 600; }}
  .entropy-mid {{ color: #fd7e14; font-weight: 600; }}
  .entropy-high {{ color: #dc3545; font-weight: 600; }}
  .title-cell {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis;
                 white-space: nowrap; color: #555; }}
  .level-bar {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .level-label {{ width: 60px; font-size: 13px; font-weight: 600; color: #555; }}
  .bar-track {{ flex: 1; background: #f0f0f0; border-radius: 6px; height: 24px;
                overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; display: flex; align-items: center;
               padding-left: 8px; font-size: 12px; color: white; font-weight: 600;
               transition: width 0.8s ease; }}
  .note {{ font-size: 12px; color: #aaa; margin-top: 8px; }}
  .footer {{ text-align: center; padding: 24px; color: #aaa; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 600; }}
  .badge-blue {{ background: #e3f2fd; color: #1565c0; }}
  .badge-green {{ background: #e8f5e9; color: #2e7d32; }}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 GraphRAG Improved — 运行报告</h1>
  <p>带结构熵惩罚的层次化 Leiden 聚类 &nbsp;|&nbsp; 生成时间：{now}</p>
</div>

<div class="container">

  <!-- 统计卡片 -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="value">{num_entities}</div>
      <div class="label">实体数量</div>
    </div>
    <div class="stat-card">
      <div class="value">{num_relations}</div>
      <div class="label">关系数量</div>
    </div>
    <div class="stat-card">
      <div class="value">{num_communities}</div>
      <div class="label">社区总数</div>
    </div>
    <div class="stat-card">
      <div class="value">{num_levels}</div>
      <div class="label">层次数量</div>
    </div>
    <div class="stat-card">
      <div class="value">{level0_entropy:.3f}</div>
      <div class="label">底层平均结构熵</div>
    </div>
    <div class="stat-card">
      <div class="value">{stats.get('elapsed_seconds', 0):.1f}s</div>
      <div class="label">运行耗时</div>
    </div>
  </div>

  <!-- 结构熵分布 -->
  <div class="section">
    <h2>📊 各层结构熵分布</h2>
    <div id="entropy-chart" class="chart-container"></div>
    <canvas id="entropyCanvas" width="1100" height="280"
            style="display:block; margin:0 auto;"></canvas>
  </div>

  <!-- 层次社区规模 -->
  <div class="section">
    <h2>🏗️ 层次社区规模</h2>
    {"".join([
        f'''<div class="level-bar">
          <div class="level-label">Level {level}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{min(100, len(df[df['level']==level])/max(1,len(df))*100*num_levels):.0f}%;
                 background: hsl({120 + level * 30}, 60%, 45%);">
              {len(df[df['level']==level])} 个社区
            </div>
          </div>
        </div>'''
        for level in sorted(df["level"].unique())
    ] if not df.empty else ["<p>暂无数据</p>"])}
  </div>

  <!-- 知识图谱可视化 -->
  <div class="section">
    <h2>🕸️ 知识图谱（Level 0 社区着色）</h2>
    <p style="font-size:13px;color:#888;margin-bottom:12px;">
      节点颜色代表所属社区，边粗细代表共现权重。拖拽节点可交互。
    </p>
    <canvas id="graphCanvas" width="1100" height="500"
            style="display:block;margin:0 auto;border:1px solid #eee;border-radius:8px;background:#fafafa;cursor:grab;"></canvas>
  </div>

  <!-- 社区详情表 -->
  <div class="section">
    <h2>📋 社区详情</h2>
    {community_table}
  </div>

</div>

<div class="footer">
  GraphRAG Improved &nbsp;|&nbsp; 结构熵约束 Leiden 聚类原型
</div>

<script>
// 绘制结构熵分布折线图
const chartData = {entropy_chart_data};
// 知识图谱数据
const graphData = {graph_data};

(function() {{
  const canvas = document.getElementById('entropyCanvas');
  if (!canvas || !chartData.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = {{ top: 20, right: 40, bottom: 50, left: 60 }};
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  // 收集所有熵值
  const allEntropies = chartData.flatMap(d => d.entropies);
  const maxH = Math.max(...allEntropies, 0.01);
  const levels = chartData.map(d => d.level);

  // 背景
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, W, H);

  // 网格线
  ctx.strokeStyle = '#e8e8e8';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {{
    const y = PAD.top + plotH - (i / 5) * plotH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#999'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((maxH * i / 5).toFixed(3), PAD.left - 8, y + 4);
  }}

  // 散点（每个社区一个点）
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'];
  chartData.forEach((levelData, li) => {{
    const x = PAD.left + (li / Math.max(1, chartData.length - 1)) * plotW;
    const color = colors[li % colors.length];

    levelData.entropies.forEach(e => {{
      const y = PAD.top + plotH - (e / maxH) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color + '88';
      ctx.fill();
    }});

    // 均值点
    const meanY = PAD.top + plotH - (levelData.mean / maxH) * plotH;
    ctx.beginPath(); ctx.arc(x, meanY, 6, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();

    // X 轴标签
    ctx.fillStyle = '#555'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Level ' + levelData.level, x, PAD.top + plotH + 20);
    ctx.fillStyle = '#888'; ctx.font = '10px sans-serif';
    ctx.fillText('λ=' + levelData.lambda, x, PAD.top + plotH + 36);
  }});

  // 均值连线
  if (chartData.length > 1) {{
    ctx.beginPath();
    ctx.strokeStyle = '#333'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
    chartData.forEach((d, i) => {{
      const x = PAD.left + (i / Math.max(1, chartData.length - 1)) * plotW;
      const y = PAD.top + plotH - (d.mean / maxH) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke(); ctx.setLineDash([]);
  }}

  // 轴标签
  ctx.fillStyle = '#555'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('聚类层级', PAD.left + plotW / 2, H - 4);
  ctx.save(); ctx.translate(14, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2); ctx.fillText('结构熵 H', 0, 0); ctx.restore();

  // 图例
  ctx.fillStyle = '#333'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.beginPath(); ctx.arc(PAD.left + plotW - 120, PAD.top + 12, 6, 0, Math.PI*2);
  ctx.fillStyle = '#555'; ctx.fill();
  ctx.fillStyle = '#555'; ctx.fillText('均值', PAD.left + plotW - 108, PAD.top + 16);
  ctx.beginPath(); ctx.arc(PAD.left + plotW - 60, PAD.top + 12, 3, 0, Math.PI*2);
  ctx.fillStyle = '#55555588'; ctx.fill();
  ctx.fillStyle = '#555'; ctx.fillText('各社区', PAD.left + plotW - 48, PAD.top + 16);
}})();

// ============================================================
// Force-directed 知识图谱可视化
// ============================================================
(function() {{
  const canvas = document.getElementById('graphCanvas');
  if (!canvas || !graphData.nodes.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // 社区颜色映射
  const commColors = [
    '#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
    '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4',
    '#8bc34a','#ff5722','#607d8b','#795548','#ff9800',
  ];
  const getColor = (commId) => {{
    if (commId < 0) return '#aaa';
    return commColors[commId % commColors.length];
  }};

  // 初始化节点位置（随机分布在画布中央区域）
  const nodes = graphData.nodes.map((n, i) => ({{
    ...n,
    x: W/2 + (Math.random() - 0.5) * W * 0.6,
    y: H/2 + (Math.random() - 0.5) * H * 0.6,
    vx: 0, vy: 0,
    r: 7,
  }}));
  const nodeMap = {{}};
  nodes.forEach(n => nodeMap[n.id] = n);

  const links = graphData.links.map(l => ({{
    source: nodeMap[l.source],
    target: nodeMap[l.target],
    weight: l.weight,
  }})).filter(l => l.source && l.target);

  // Force simulation 参数
  const REPULSION = 800;
  const ATTRACTION = 0.04;
  const DAMPING = 0.85;
  const CENTER_PULL = 0.01;
  let running = true;

  function simulate() {{
    // 斥力（节点间）
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;
        const force = REPULSION / (dist * dist);
        const fx = force * dx / dist;
        const fy = force * dy / dist;
        nodes[i].vx -= fx; nodes[i].vy -= fy;
        nodes[j].vx += fx; nodes[j].vy += fy;
      }}
    }}
    // 引力（边）
    links.forEach(l => {{
      const dx = l.target.x - l.source.x;
      const dy = l.target.y - l.source.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const force = ATTRACTION * dist * Math.log(1 + l.weight);
      l.source.vx += force * dx / dist;
      l.source.vy += force * dy / dist;
      l.target.vx -= force * dx / dist;
      l.target.vy -= force * dy / dist;
    }});
    // 中心引力 + 阻尼 + 边界
    nodes.forEach(n => {{
      n.vx += (W/2 - n.x) * CENTER_PULL;
      n.vy += (H/2 - n.y) * CENTER_PULL;
      n.vx *= DAMPING; n.vy *= DAMPING;
      n.x = Math.max(20, Math.min(W-20, n.x + n.vx));
      n.y = Math.max(20, Math.min(H-20, n.y + n.vy));
    }});
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, W, H);

    // 绘制边
    links.forEach(l => {{
      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);
      ctx.strokeStyle = 'rgba(150,150,150,0.35)';
      ctx.lineWidth = Math.min(3, 0.5 + l.weight * 0.3);
      ctx.stroke();
    }});

    // 绘制节点
    nodes.forEach(n => {{
      const color = getColor(n.community);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 节点标签（只显示较短的名称）
      if (n.id.length <= 15) {{
        ctx.fillStyle = '#333';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(n.id, n.x, n.y + n.r + 11);
      }}
    }});
  }}

  // 动画循环（前 120 帧模拟，之后静止）
  let frame = 0;
  function loop() {{
    if (frame < 120) {{ simulate(); frame++; }}
    draw();
    if (frame < 120) requestAnimationFrame(loop);
  }}
  loop();

  // 拖拽交互
  let dragging = null;
  canvas.addEventListener('mousedown', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    dragging = nodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 4) || null;
  }});
  canvas.addEventListener('mousemove', e => {{
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    dragging.x = (e.clientX - rect.left) * (W / rect.width);
    dragging.y = (e.clientY - rect.top) * (H / rect.height);
    dragging.vx = 0; dragging.vy = 0;
    simulate(); draw();
  }});
  canvas.addEventListener('mouseup', () => {{ dragging = null; }});
}})();
</script>

</body>
</html>"""

    path = str(out / "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return path


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def save_results(
    result: PipelineResult,
    config: OutputConfig,
) -> Dict[str, str]:
    """
    根据配置将所有结果持久化，返回 {文件类型: 路径} 映射。

    Parameters
    ----------
    result : PipelineResult
        Pipeline 运行结果
    config : OutputConfig
        输出配置

    Returns
    -------
    Dict[str, str]
        {描述: 文件路径} 映射，供主程序打印输出路径
    """
    output_dir = config.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}

    if config.console_summary:
        summary = print_console_summary(result)
        summary_path = str(Path(output_dir) / "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        saved["summary"] = summary_path

    if config.csv_export:
        csv_files = export_csv(result, output_dir)
        saved.update(csv_files)

    if config.parquet_export:
        try:
            parquet_files = export_parquet(result, output_dir)
            saved.update(parquet_files)
        except ImportError:
            print("  [提示] 导出 Parquet 需要安装 pyarrow：pip install pyarrow")

    if config.html_report:
        html_path = export_html_report(result, output_dir)
        saved["report.html"] = html_path

    return saved
