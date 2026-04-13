"""
experiments/analyze_results.py
--------------------------------
读取 comparison_results.json，生成详细的分析报告和可视化 HTML。

用法：
  python -m graphrag_improved.experiments.analyze_results
  python -m graphrag_improved.experiments.analyze_results --results-dir ./experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_results(results_dir: str) -> dict:
    path = Path(results_dir) / "comparison_results.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_analysis(results: dict) -> None:
    """打印详细分析。"""
    b = results["baseline"]
    o = results["ours"]
    imp = results.get("improvement", {})

    print("\n" + "="*65)
    print("  GraphRAG Improved — 实验结果深度分析")
    print("="*65)

    print("\n【一、实验规模】")
    print(f"  实体数量    : {b['num_entities']} 个")
    print(f"  关系数量    : {b['num_relationships']} 条")

    print("\n【二、社区结构对比】")
    print(f"  {'指标':<20} {'Baseline':>12} {'Ours':>12} {'变化':>10}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    print(f"  {'社区总数':<20} {b['num_communities']:>12} {o['num_communities']:>12}")
    print(f"  {'层次数量':<20} {b['num_levels']:>12} {o['num_levels']:>12}")
    print(f"  {'平均社区大小':<20} {b['avg_community_size']:>12.2f} {o['avg_community_size']:>12.2f}")
    print(f"  {'模块度 Q':<20} {b['modularity']:>12.4f} {o['modularity']:>12.4f}")
    print(f"  {'平均结构熵':<20} {b['avg_structural_entropy']:>12.4f} {o['avg_structural_entropy']:>12.4f}  {imp.get('avg_structural_entropy','')}")
    print(f"  {'Level0 纯净率':<20} {b['level0_purity_rate']:>11.2%} {o['level0_purity_rate']:>11.2%}  {imp.get('level0_purity_rate','')}")

    # 各层结构熵
    b_entropy = b.get("entropy_by_level", {})
    o_entropy = o.get("entropy_by_level", {})
    if b_entropy or o_entropy:
        all_levels = sorted(set(list(b_entropy.keys()) + list(o_entropy.keys())), key=int)
        print(f"\n  各层平均结构熵（λ 退火效果验证）：")
        print(f"  {'层级':<10} {'Baseline':>12} {'Ours':>12}")
        for lv in all_levels:
            be = b_entropy.get(lv, 0)
            oe = o_entropy.get(lv, 0)
            print(f"  Level {lv:<4}  {be:>12.4f} {oe:>12.4f}")

    print("\n【三、检索质量对比（MultiHop-RAG）】")
    print(f"  {'指标':<20} {'Baseline':>12} {'Ours':>12} {'提升':>10}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    metrics = [
        ("MRR", "mrr", "mrr"),
        ("Precision@1", "precision_at_1", ""),
        ("Precision@3", "precision_at_3", ""),
        ("Precision@5", "precision_at_5", "precision_at_5"),
        ("Precision@10", "precision_at_10", ""),
        ("Recall@5", "recall_at_5", ""),
        ("Recall@10", "recall_at_10", "recall_at_10"),
        ("F1@5", "f1_at_5", ""),
        ("NDCG@5", "ndcg_at_5", ""),
        ("NDCG@10", "ndcg_at_10", "ndcg_at_10"),
    ]
    for label, key, imp_key in metrics:
        bv = b.get(key, 0)
        ov = o.get(key, 0)
        delta = imp.get(imp_key, "") if imp_key else ""
        print(f"  {label:<20} {bv:>12.4f} {ov:>12.4f} {delta:>10}")

    print("\n【四、结论】")
    mrr_b = b.get("mrr", 0)
    mrr_o = o.get("mrr", 0)
    ndcg_b = b.get("ndcg_at_10", 0)
    ndcg_o = o.get("ndcg_at_10", 0)
    purity_b = b.get("level0_purity_rate", 0)
    purity_o = o.get("level0_purity_rate", 0)

    if mrr_o > mrr_b:
        print(f"  ✅ MRR 提升 {(mrr_o-mrr_b)/mrr_b*100:.1f}%，结构熵约束改善了检索排序质量")
    elif mrr_o == mrr_b:
        print(f"  ⚠️  MRR 持平，检索排序质量无变化（可能数据规模不足）")
    else:
        print(f"  ❌ MRR 下降 {(mrr_b-mrr_o)/mrr_b*100:.1f}%，需要调整 λ 参数")

    if ndcg_o > ndcg_b:
        print(f"  ✅ NDCG@10 提升 {(ndcg_o-ndcg_b)/ndcg_b*100:.1f}%，相关文档排名更靠前")
    elif ndcg_o == ndcg_b:
        print(f"  ⚠️  NDCG@10 持平")
    else:
        print(f"  ❌ NDCG@10 下降")

    if purity_o > purity_b:
        print(f"  ✅ Level0 纯净率 {purity_b:.1%} → {purity_o:.1%}，物理边界保持效果显著")
    else:
        print(f"  ⚠️  Level0 纯净率无提升（λ 可能需要调大）")

    print("="*65)


def generate_html_report(results: dict, output_path: str) -> None:
    """生成 HTML 格式的对比报告。"""
    b = results["baseline"]
    o = results["ours"]
    imp = results.get("improvement", {})

    # 各层结构熵数据（用于折线图）
    b_entropy = b.get("entropy_by_level", {})
    o_entropy = o.get("entropy_by_level", {})
    all_levels = sorted(set(list(b_entropy.keys()) + list(o_entropy.keys())), key=int)
    entropy_labels = [f"Level {lv}" for lv in all_levels]
    entropy_baseline = [b_entropy.get(lv, 0) for lv in all_levels]
    entropy_ours = [o_entropy.get(lv, 0) for lv in all_levels]

    # 检索指标对比数据（用于柱状图）
    retrieval_labels = ["MRR", "P@1", "P@3", "P@5", "P@10", "R@5", "R@10", "NDCG@5", "NDCG@10"]
    retrieval_keys = ["mrr", "precision_at_1", "precision_at_3", "precision_at_5",
                      "precision_at_10", "recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10"]
    retrieval_baseline = [b.get(k, 0) for k in retrieval_keys]
    retrieval_ours = [o.get(k, 0) for k in retrieval_keys]

    import json as _json
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GraphRAG Improved — 实验结果对比</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f7fa; color: #333; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             color: white; padding: 40px; }}
  .header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }}
  .header p {{ opacity: 0.75; font-size: 14px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .section {{ background: white; border-radius: 12px; padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
  .section h2 {{ font-size: 17px; font-weight: 600; margin-bottom: 16px;
                 padding-bottom: 10px; border-bottom: 2px solid #f0f0f0; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                   gap: 12px; margin-bottom: 16px; }}
  .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 14px; text-align: center; }}
  .metric-card .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
  .metric-card .values {{ display: flex; justify-content: center; gap: 12px; }}
  .metric-card .val {{ font-size: 20px; font-weight: 700; }}
  .val.baseline {{ color: #6c757d; }}
  .val.ours {{ color: #0f3460; }}
  .val.better {{ color: #28a745; }}
  .val.worse {{ color: #dc3545; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; margin-top: 4px; }}
  .badge-up {{ background: #d4edda; color: #155724; }}
  .badge-down {{ background: #f8d7da; color: #721c24; }}
  .badge-flat {{ background: #e2e3e5; color: #383d41; }}
  canvas {{ display: block; margin: 0 auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fa; padding: 8px 12px; text-align: left;
        font-weight: 600; border-bottom: 2px solid #e9ecef; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #f8f9fa; }}
  .better-cell {{ color: #28a745; font-weight: 600; }}
  .worse-cell {{ color: #dc3545; font-weight: 600; }}
</style>
</head>
<body>
<div class="header">
  <h1>🔬 GraphRAG Improved — 实验结果对比报告</h1>
  <p>MultiHop-RAG 数据集 &nbsp;|&nbsp; Baseline (λ=0) vs 结构熵约束 (λ=1000)</p>
</div>
<div class="container">

  <!-- 核心指标卡片 -->
  <div class="section">
    <h2>📊 核心检索指标</h2>
    <div class="metrics-grid">
      {_metric_card("MRR", b.get("mrr",0), o.get("mrr",0))}
      {_metric_card("Precision@5", b.get("precision_at_5",0), o.get("precision_at_5",0))}
      {_metric_card("Recall@10", b.get("recall_at_10",0), o.get("recall_at_10",0))}
      {_metric_card("NDCG@10", b.get("ndcg_at_10",0), o.get("ndcg_at_10",0))}
      {_metric_card("Level0 纯净率", b.get("level0_purity_rate",0), o.get("level0_purity_rate",0), is_pct=True)}
      {_metric_card("平均结构熵", b.get("avg_structural_entropy",0), o.get("avg_structural_entropy",0), lower_better=True)}
    </div>
  </div>

  <!-- 检索指标柱状图 -->
  <div class="section">
    <h2>📈 检索指标全览</h2>
    <canvas id="retrievalChart" width="1000" height="300"></canvas>
  </div>

  <!-- 结构熵折线图 -->
  <div class="section">
    <h2>🔥 各层结构熵分布（λ 退火效果）</h2>
    <canvas id="entropyChart" width="1000" height="260"></canvas>
    <p style="font-size:12px;color:#888;margin-top:8px;text-align:center;">
      Ours 底层熵更低（物理边界保持），高层熵更高（跨文档语义融合）
    </p>
  </div>

  <!-- 详细数据表 -->
  <div class="section">
    <h2>📋 完整指标数据</h2>
    <table>
      <thead>
        <tr><th>指标</th><th>Baseline (λ=0)</th><th>Ours (λ=1000)</th><th>提升</th></tr>
      </thead>
      <tbody>
        {_table_rows(b, o, imp)}
      </tbody>
    </table>
  </div>

</div>

<script>
const retrievalLabels = {_json.dumps(retrieval_labels)};
const retrievalBaseline = {_json.dumps(retrieval_baseline)};
const retrievalOurs = {_json.dumps(retrieval_ours)};
const entropyLabels = {_json.dumps(entropy_labels)};
const entropyBaseline = {_json.dumps(entropy_baseline)};
const entropyOurs = {_json.dumps(entropy_ours)};

// 检索指标柱状图
(function() {{
  const canvas = document.getElementById('retrievalChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = {{ top: 20, right: 20, bottom: 50, left: 50 }};
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const n = retrievalLabels.length;
  const groupW = plotW / n;
  const barW = groupW * 0.35;

  ctx.fillStyle = '#fafafa'; ctx.fillRect(0, 0, W, H);

  // 网格
  for (let i = 0; i <= 5; i++) {{
    const y = PAD.top + plotH - (i / 5) * plotH;
    ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + plotW, y); ctx.stroke();
    ctx.fillStyle = '#999'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((i / 5).toFixed(1), PAD.left - 5, y + 3);
  }}

  retrievalLabels.forEach((label, i) => {{
    const cx = PAD.left + (i + 0.5) * groupW;
    const bH = retrievalBaseline[i] * plotH;
    const oH = retrievalOurs[i] * plotH;

    // Baseline 柱
    ctx.fillStyle = '#6c757d88';
    ctx.fillRect(cx - barW - 2, PAD.top + plotH - bH, barW, bH);
    // Ours 柱
    const oColor = retrievalOurs[i] >= retrievalBaseline[i] ? '#0f346088' : '#dc354588';
    ctx.fillStyle = oColor;
    ctx.fillRect(cx + 2, PAD.top + plotH - oH, barW, oH);

    // X 轴标签
    ctx.fillStyle = '#555'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(label, cx, PAD.top + plotH + 18);
  }});

  // 图例
  ctx.fillStyle = '#6c757d'; ctx.fillRect(PAD.left, 5, 12, 10);
  ctx.fillStyle = '#555'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Baseline', PAD.left + 16, 14);
  ctx.fillStyle = '#0f3460'; ctx.fillRect(PAD.left + 80, 5, 12, 10);
  ctx.fillText('Ours', PAD.left + 96, 14);
}})();

// 结构熵折线图
(function() {{
  const canvas = document.getElementById('entropyChart');
  if (!canvas || !entropyLabels.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = {{ top: 20, right: 30, bottom: 45, left: 55 }};
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const n = entropyLabels.length;

  const allVals = [...entropyBaseline, ...entropyOurs];
  const maxV = Math.max(...allVals, 0.01);

  ctx.fillStyle = '#fafafa'; ctx.fillRect(0, 0, W, H);

  // 网格
  for (let i = 0; i <= 4; i++) {{
    const y = PAD.top + plotH - (i / 4) * plotH;
    ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + plotW, y); ctx.stroke();
    ctx.fillStyle = '#999'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((maxV * i / 4).toFixed(3), PAD.left - 5, y + 3);
  }}

  const xPos = (i) => PAD.left + (n <= 1 ? plotW / 2 : i / (n - 1) * plotW);
  const yPos = (v) => PAD.top + plotH - (v / maxV) * plotH;

  // Baseline 折线
  ctx.beginPath(); ctx.strokeStyle = '#6c757d'; ctx.lineWidth = 2.5;
  entropyBaseline.forEach((v, i) => i === 0 ? ctx.moveTo(xPos(i), yPos(v)) : ctx.lineTo(xPos(i), yPos(v)));
  ctx.stroke();
  entropyBaseline.forEach((v, i) => {{
    ctx.beginPath(); ctx.arc(xPos(i), yPos(v), 5, 0, Math.PI*2);
    ctx.fillStyle = '#6c757d'; ctx.fill();
  }});

  // Ours 折线
  ctx.beginPath(); ctx.strokeStyle = '#0f3460'; ctx.lineWidth = 2.5;
  entropyOurs.forEach((v, i) => i === 0 ? ctx.moveTo(xPos(i), yPos(v)) : ctx.lineTo(xPos(i), yPos(v)));
  ctx.stroke();
  entropyOurs.forEach((v, i) => {{
    ctx.beginPath(); ctx.arc(xPos(i), yPos(v), 5, 0, Math.PI*2);
    ctx.fillStyle = '#0f3460'; ctx.fill();
  }});

  // X 轴标签
  entropyLabels.forEach((label, i) => {{
    ctx.fillStyle = '#555'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(label, xPos(i), PAD.top + plotH + 18);
  }});

  // 图例
  ctx.strokeStyle = '#6c757d'; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(PAD.left, 12); ctx.lineTo(PAD.left + 20, 12); ctx.stroke();
  ctx.fillStyle = '#555'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Baseline (λ=0)', PAD.left + 24, 16);
  ctx.strokeStyle = '#0f3460';
  ctx.beginPath(); ctx.moveTo(PAD.left + 120, 12); ctx.lineTo(PAD.left + 140, 12); ctx.stroke();
  ctx.fillText('Ours (λ=1000)', PAD.left + 144, 16);
}})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告已保存：{output_path}")


def _metric_card(label: str, baseline: float, ours: float,
                 is_pct: bool = False, lower_better: bool = False) -> str:
    fmt = lambda v: f"{v:.1%}" if is_pct else f"{v:.4f}"
    better = ours < baseline if lower_better else ours > baseline
    equal = abs(ours - baseline) < 1e-6
    ours_class = "flat" if equal else ("better" if better else "worse")
    if not equal:
        delta = (ours - baseline) / (baseline + 1e-9) * 100
        sign = "+" if delta >= 0 else ""
        badge_class = "badge-up" if better else "badge-down"
        badge = f'<span class="badge {badge_class}">{sign}{delta:.1f}%</span>'
    else:
        badge = '<span class="badge badge-flat">持平</span>'
    return f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="values">
        <span class="val baseline">{fmt(baseline)}</span>
        <span>→</span>
        <span class="val {ours_class}">{fmt(ours)}</span>
      </div>
      {badge}
    </div>"""


def _table_rows(b: dict, o: dict, imp: dict) -> str:
    rows_def = [
        ("MRR", "mrr"),
        ("Precision@1", "precision_at_1"),
        ("Precision@3", "precision_at_3"),
        ("Precision@5", "precision_at_5"),
        ("Precision@10", "precision_at_10"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("F1@5", "f1_at_5"),
        ("NDCG@5", "ndcg_at_5"),
        ("NDCG@10", "ndcg_at_10"),
        ("模块度 Q", "modularity"),
        ("平均结构熵", "avg_structural_entropy"),
        ("Level0 纯净率", "level0_purity_rate"),
        ("社区数量", "num_communities"),
        ("层次数量", "num_levels"),
    ]
    html = ""
    for label, key in rows_def:
        bv = b.get(key, 0)
        ov = o.get(key, 0)
        delta = imp.get(key, "")
        is_pct = "purity" in key
        fmt = lambda v: f"{v:.2%}" if is_pct else (f"{v}" if isinstance(v, int) else f"{v:.4f}")
        better = ov > bv
        cell_class = "better-cell" if better and bv != ov else ("worse-cell" if not better and bv != ov else "")
        html += f"<tr><td>{label}</td><td>{fmt(bv)}</td><td class='{cell_class}'>{fmt(ov)}</td><td>{delta}</td></tr>\n"
    return html


def main():
    parser = argparse.ArgumentParser(description="分析实验结果")
    parser.add_argument("--results-dir", default="./experiments/results", help="结果目录")
    parser.add_argument("--html", action="store_true", help="生成 HTML 报告")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # 优先找真实实验结果，其次找 smoke test 结果
    for candidate in [results_dir, Path("./experiments/results_smoke")]:
        result_file = candidate / "comparison_results.json"
        if result_file.exists():
            results_dir = candidate
            break
    else:
        print(f"[错误] 找不到结果文件，请先运行实验：")
        print(f"  python -m graphrag_improved.experiments.smoke_test")
        sys.exit(1)

    print(f"\n[加载] {results_dir / 'comparison_results.json'}")
    results = load_results(str(results_dir))
    print_analysis(results)

    if args.html:
        html_path = str(results_dir / "comparison_report.html")
        generate_html_report(results, html_path)
    else:
        # 默认也生成 HTML
        html_path = str(results_dir / "comparison_report.html")
        generate_html_report(results, html_path)
        print(f"\n  用浏览器打开查看可视化报告：{html_path}")


if __name__ == "__main__":
    main()
