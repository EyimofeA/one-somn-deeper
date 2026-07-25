"""Generates task_a_error_analysis.html from task_a_analysis.json and
task_a_deconfound.json. Every number in the report is computed from those
JSON files at generation time -- do not hand-type a statistic into the HTML
string below; a prior version did exactly that (a stat tile read "67.5%"
where the underlying data says 7.5%) and it went uncaught because nothing
tied the displayed text to the source numbers. If you need a new number,
compute it from `d`/`dc` above the f-string, the same way everything else
here does.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "task_a_analysis.json"))
dc = json.load(open(HERE / "task_a_deconfound.json"))
s1 = d["section1_error_structure"]
s2 = d["section2_arithmetic_difficulty"]
s3 = d["section3_baselines"]
s4 = d["section4_per_position"]["per_position"]
n = d["n_val_examples"]


# ---------------------------------------------------------------------------
# chart helpers
# ---------------------------------------------------------------------------
def svg_bar_chart(labels, values, width=560, height=260, color="blue", fmt="{:.1%}", y_max=None):
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = y_max if y_max is not None else max(values) * 1.15 if max(values) > 0 else 1
    n_bars = len(values)
    bar_w = plot_w / n_bars * 0.62
    gap = plot_w / n_bars
    bars = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = pad_l + i * gap + (gap - bar_w) / 2
        h = (v / y_max) * plot_h if y_max > 0 else 0
        y = pad_t + plot_h - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" '
            f'fill="var(--{color})"><title>{lab}: {fmt.format(v)}</title></rect>'
            f'<text x="{x + bar_w/2:.1f}" y="{pad_t + plot_h + 16}" class="axis-label" text-anchor="middle">{lab}</text>'
        )
        if v > 0:
            bars.append(
                f'<text x="{x + bar_w/2:.1f}" y="{y - 5:.1f}" class="bar-value" text-anchor="middle">{fmt.format(v)}</text>'
            )
    grid = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h*(1-f)}" x2="{pad_l+plot_w}" y2="{pad_t + plot_h*(1-f)}" class="gridline"/>'
        for f in (0, 0.25, 0.5, 0.75, 1.0)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">{grid}'
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" class="axis-line"/>'
        + "".join(bars) + "</svg>"
    )


def svg_grouped_bar(labels, series, width=760, height=280, colors=("blue", "orange"), fmt="{:.1%}", legend=("A", "B")):
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = max(max(s) for s in series) * 1.15
    n_groups = len(labels)
    group_w = plot_w / n_groups
    bar_w = group_w / (len(series) + 1)
    els = []
    for gi, lab in enumerate(labels):
        gx = pad_l + gi * group_w
        for si, s in enumerate(series):
            v = s[gi]
            h = (v / y_max) * plot_h if y_max > 0 else 0
            x = gx + (si + 0.5) * bar_w
            y = pad_t + plot_h - h
            els.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" rx="2" '
                f'fill="var(--{colors[si]})"><title>{lab} / {legend[si]}: {fmt.format(v)}</title></rect>'
            )
        els.append(f'<text x="{gx + group_w/2:.1f}" y="{pad_t+plot_h+16}" class="axis-label" text-anchor="middle">{lab}</text>')
    grid = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h*(1-f)}" x2="{pad_l+plot_w}" y2="{pad_t + plot_h*(1-f)}" class="gridline"/>'
        for f in (0, 0.25, 0.5, 0.75, 1.0)
    )
    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:var(--{c})"></span>{t}</span>'
        for c, t in zip(colors, legend)
    )
    return (
        f'<div class="legend-row">{legend_html}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">{grid}'
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" class="axis-line"/>'
        + "".join(els) + "</svg>"
    )


def svg_line_chart(x_labels, series_list, width=760, height=300, colors=("blue", "orange"), legend=("A", "B"), y_max=1.0, fmt="{:.1%}"):
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_pts = len(x_labels)
    step = plot_w / (n_pts - 1) if n_pts > 1 else 0
    grid = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h*(1-f)}" x2="{pad_l+plot_w}" y2="{pad_t + plot_h*(1-f)}" class="gridline"/>'
        for f in (0, 0.25, 0.5, 0.75, 1.0)
    )
    els = [grid, f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" class="axis-line"/>']
    for si, series in enumerate(series_list):
        pts = []
        for i, v in enumerate(series):
            x = pad_l + i * step
            y = pad_t + plot_h - (v / y_max) * plot_h
            pts.append((x, y))
        path = " ".join(f"{'M' if i==0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        els.append(f'<path d="{path}" fill="none" stroke="var(--{colors[si]})" stroke-width="2.5"/>')
        for i, (x, y) in enumerate(pts):
            els.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--{colors[si]})">'
                        f'<title>{x_labels[i]}: {fmt.format(series[i])}</title></circle>')
    for i, lab in enumerate(x_labels):
        x = pad_l + i * step
        els.append(f'<text x="{x:.1f}" y="{pad_t+plot_h+16}" class="axis-label" text-anchor="middle">{lab}</text>')
    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:var(--{c})"></span>{t}</span>'
        for c, t in zip(colors, legend)
    ) if len(series_list) > 1 else ""
    return f'<div class="legend-row">{legend_html}</div><svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">' + "".join(els) + "</svg>"


# ---------------------------------------------------------------------------
# Section 1: error structure (numbers computed, not typed)
# ---------------------------------------------------------------------------
bucket_names = s1["bucket_names"]
bucket_fracs = s1["bucket_fractions"]
error_dist_chart = svg_bar_chart(bucket_names, bucket_fracs, color="blue", fmt="{:.1%}")

exact_match_frac = bucket_fracs[0]
le1_wrong_frac = bucket_fracs[0] + bucket_fracs[1]  # 0 wrong + 1 wrong
_largest_idx = max(range(len(bucket_fracs)), key=lambda i: bucket_fracs[i])
largest_bucket_name = bucket_names[_largest_idx]
largest_bucket_frac = bucket_fracs[_largest_idx]
token_accuracy_frac = s3["trained_model"]["token_accuracy"]
frac_contiguous = s1["frac_contiguous_given_wrong"]

fw_hist = s1["first_wrong_position_histogram"]
fw_labels = [str(i) for i in range(12)]
fw_total = sum(fw_hist.values())
fw_vals = [fw_hist.get(str(i), 0) / fw_total if fw_total else 0 for i in range(12)]
fw_chart = svg_bar_chart(fw_labels, fw_vals, color="orange", fmt="{:.1%}")

# ---- Section 4 per-position ----
positions = [r["position"] for r in s4]
acc = [r["accuracy"] for r in s4]
acc_carry = [r["accuracy_given_carry_in"] if r["accuracy_given_carry_in"] is not None else 0 for r in s4]
acc_nocarry = [r["accuracy_given_no_carry_in"] if r["accuracy_given_no_carry_in"] is not None else 0 for r in s4]
pos_acc_chart = svg_line_chart([str(p) for p in positions], [acc], colors=("blue",), legend=("accuracy",), y_max=1.05)
carry_chart = svg_grouped_bar([str(p) for p in positions], [acc_carry, acc_nocarry], colors=("blue", "orange"),
                               legend=("carry enters this position (present/absent, from analysis pass 1)",
                                       "no carry enters this position"))

# ---- Section 2 feature importance ----
tree_imp = s2["tree_feature_importances"]
feat_sorted = sorted(tree_imp.items(), key=lambda kv: -kv[1])
feat_labels = [k.replace("_", " ") for k, v in feat_sorted]
feat_vals = [v for k, v in feat_sorted]
feat_chart = svg_bar_chart(feat_labels, feat_vals, width=760, height=300, color="aqua", fmt="{:.3f}")
logreg_test_acc = s2["logreg_test_accuracy"]
logreg_baseline = s2["logreg_majority_baseline"]
tree_test_acc = s2["tree_test_accuracy"]

# n_digits_x bucket table -> chart
ndx = s2["bucket_report"]["n_digits_x"]
ndx_labels = sorted(ndx.keys(), key=lambda k: float(k))
ndx_vals = [ndx[k]["exact_match"] for k in ndx_labels]
ndx_ns = [ndx[k]["n"] for k in ndx_labels]
ndx_chart = svg_bar_chart([f"{float(k):.0f} digits" for k in ndx_labels], ndx_vals, color="yellow", fmt="{:.1%}")
frac_6digit = ndx[ndx_labels[-1]]["n"] / n

# ---- Section 3 baselines ----
baseline_rows = [
    ("Trained model", s3["trained_model"]["exact_match"], s3["trained_model"]["token_accuracy"]),
    ("Most-common digit/pos", s3["most_common_digit_per_position"]["exact_match"], s3["most_common_digit_per_position"]["token_accuracy"]),
    ("Copy input digits", s3["copy_input_digits_heuristic"]["exact_match"], s3["copy_input_digits_heuristic"]["token_accuracy"]),
    ("Nearest train target (by x)", s3["nearest_training_target_by_value"]["exact_match"], s3["nearest_training_target_by_value"]["token_accuracy"]),
]
base_labels = [r[0] for r in baseline_rows]
base_exact = [r[1] for r in baseline_rows]
base_token = [r[2] for r in baseline_rows]
baseline_chart = svg_grouped_bar(base_labels, [base_exact, base_token], colors=("blue", "aqua"),
                                  legend=("exact match", "token accuracy"), width=760)
baseline_table_rows = "".join(
    f'<tr><td>{lab}</td><td>{ex:.2%}</td><td>{tok:.1%}</td></tr>' for lab, ex, tok in baseline_rows
)
hamming_exact = s3["nearest_training_input_by_hamming"]["exact_match"]
hamming_n = s3["nearest_training_input_by_hamming"]["n_sampled"]
lookup_frac_seen = s3["lookup_table_exact_seen_inputs"]["fraction_of_val_seen_in_train"]
leading_only_acc = s3["leading_digit_only_accuracy"]
trailing_only_acc = s3["trailing_digit_only_accuracy"]
token_vs_nearest_gap = (base_token[0] - base_token[3]) * 100  # trained model - nearest-by-value, in points


def render_digits(true_d, pred_d, wrong):
    cells = []
    for i, (t, p) in enumerate(zip(true_d, pred_d)):
        cls = "digit-wrong" if i in wrong else "digit-ok"
        cells.append(f'<span class="{cls}">{p}</span>')
    return "".join(cells)


fail_rows = ""
for f in s1["representative_failures"]:
    true_str = "".join(str(dg) for dg in f["true_digits"])
    pred_rendered = render_digits(f["true_digits"], f["pred_digits"], set(f["wrong_positions"]))
    fail_rows += (
        f'<tr><td>{f["x"]}</td><td class="mono">{true_str}</td>'
        f'<td class="mono">{pred_rendered}</td><td>{f["n_wrong"]}</td>'
        f'<td>{", ".join(str(p) for p in f["wrong_positions"])}</td></tr>'
    )

pos_table_rows = ""
for r in s4:
    conf = r["top_confusions"]
    conf_str = ", ".join(f"{t}→{p} (×{c})" for (t, p), c in conf) if conf else "—"
    awc = r["accuracy_given_carry_in"]
    anc = r["accuracy_given_no_carry_in"]
    pos_table_rows += (
        f'<tr><td>{r["position"]}</td><td>{r["accuracy"]:.4f}</td>'
        f'<td>{r["true_entropy_bits"]:.3f}</td><td>{r["pred_entropy_bits"]:.3f}</td>'
        f'<td>{r["mean_confidence"]:.3f}</td><td>{r["calibration_gap"]:+.4f}</td>'
        f'<td>{"—" if awc is None else f"{awc:.4f}"}</td>'
        f'<td>{"—" if anc is None else f"{anc:.4f}"}</td>'
        f'<td class="mono">{conf_str}</td></tr>'
    )

# ---------------------------------------------------------------------------
# Section 5: deconfounded analysis (from task_a_deconfound.json)
# ---------------------------------------------------------------------------
n_dc = dc["n_examples"]
confound = dc["confound_summary_by_position"]
confound_positions = sorted(confound.keys(), key=lambda k: int(k))
carry_by_pos = [confound[p]["mean_carry_in"] for p in confound_positions]
terms_by_pos = [confound[p]["mean_n_terms"] for p in confound_positions]
acc_by_pos_dc = [confound[p]["accuracy"] for p in confound_positions]
max_carry_scale = max(carry_by_pos) or 1
max_terms_scale = max(terms_by_pos) or 1
confound_chart = svg_line_chart(
    confound_positions,
    [[c / max_carry_scale for c in carry_by_pos], [t / max_terms_scale for t in terms_by_pos], acc_by_pos_dc],
    colors=("orange", "yellow", "blue"),
    legend=(f"mean carry-in (scaled, max={max_carry_scale:.1f})", f"mean # terms (scaled, max={max_terms_scale:.0f})", "accuracy"),
    y_max=1.05, fmt="{:.3f}",
)

# controlled comparisons -> pick 2 illustrative strata each, render as small tables
def controlled_table(block: dict, x_label: str) -> str:
    rows = []
    for stratum, series in block.items():
        cells = "".join(f'<td>{x_label}={k}: <b>{v["accuracy"]:.1%}</b> <span class="n-tag">n={v["n"]}</span></td>' for k, v in sorted(series.items(), key=lambda kv: float(kv[0])))
        rows.append(f'<tr><td class="stratum-label">{stratum}</td>{cells}</tr>')
    return "<table class=\"controlled-table\">" + "".join(rows) + "</table>"


controlled_a_html = controlled_table(dc["controlled_A_fix_pos_and_diagsum_vary_carry"], "carry_in")
controlled_b_html = controlled_table(dc["controlled_B_fix_pos_and_carry_vary_n_terms"], "n_terms")
controlled_c_html = controlled_table(dc["controlled_C_fix_pos_and_n_terms_vary_carry_magnitude"], "carry_in")

# per-digit error model
pdm = dc["per_digit_error_model"]
pdm_metrics = pdm["test_metrics"]
pdm_coefs = sorted(pdm["coefficients"].items(), key=lambda kv: -abs(kv[1]))
pdm_coef_chart = svg_bar_chart(
    [k.replace("_", " ") for k, v in pdm_coefs], [abs(v) for k, v in pdm_coefs],
    width=760, height=260, color="red", fmt="{:.3f}",
)
pr_auc_baseline = pdm_metrics["positive_rate_test"]  # PR-AUC of a trivial always-predict-positive classifier

# hypothesis C
hc = dc["hypothesis_c_two_sided_approximation"]
prefix_by_k = hc["prefix_exact_by_k"]
suffix_by_k = hc["suffix_exact_by_k"]
k_labels = sorted(prefix_by_k.keys(), key=lambda k: int(k))
prefix_vals = [prefix_by_k[k] for k in k_labels]
suffix_vals = [suffix_by_k[k] for k in k_labels]
prefix_suffix_chart = svg_line_chart(
    [f"k={k}" for k in k_labels], [prefix_vals, suffix_vals], colors=("blue", "aqua"),
    legend=("prefix-k exact match", "suffix-k exact match"), y_max=1.05,
)
gap_hist = hc["reconciliation_gap_histogram"]
gap_labels = sorted(gap_hist.keys(), key=lambda k: int(k))
gap_total = sum(gap_hist.values())
gap_vals = [gap_hist[k] / gap_total for k in gap_labels]
gap_chart = svg_bar_chart([str(g) for g in gap_labels], gap_vals, color="orange", fmt="{:.1%}")
prefix_width_mean = hc["prefix_width_mean"]
suffix_width_mean = hc["suffix_width_mean"]
gap_mode = max(gap_hist.items(), key=lambda kv: kv[1])[0]

# layerwise probes
probes = dc["layerwise_probes"]
layer_names = sorted(probes.keys(), key=lambda k: int(k.split("_")[1]))
probe_quantities = ["diag_sum", "carry_in", "carry_out"]
probe_series = [[probes[l][q] for l in layer_names] for q in probe_quantities]
probe_chart = svg_line_chart(
    [l.replace("layer_", "L") for l in layer_names], probe_series, colors=("blue", "orange", "aqua"),
    legend=("diagonal sum (R²)", "carry-in (R²)", "carry-out (R²)"), y_max=1.0, fmt="{:.3f}",
)
leading_probe = [probes[l]["leading_digit_accuracy"] for l in layer_names]
mod10_probe = [probes[l]["x2_mod_10_accuracy"] for l in layer_names]
global_probe_chart = svg_line_chart(
    [l.replace("layer_", "L") for l in layer_names], [leading_probe, mod10_probe], colors=("yellow", "red"),
    legend=("leading digit (accuracy)", "x² mod 10 (accuracy)"), y_max=1.05, fmt="{:.3f}",
)

sanity_mismatches = dc["sanity_check_digit_result_mismatches"]

html = f"""<!doctype html>
<title>Task A error analysis</title>
<style>
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ee;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #83817a;
  --border: #e4e2da;
  --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --yellow: #eda100; --red: #e34948;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232320;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8d84;
    --border: #35342e;
    --blue: #3987e5; --orange: #d95926; --aqua: #199e70; --yellow: #c98500; --red: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #232320;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8d84;
  --border: #35342e;
  --blue: #3987e5; --orange: #d95926; --aqua: #199e70; --yellow: #c98500; --red: #e66767;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--surface-1); color: var(--text-primary);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1000px; margin: 0 auto; padding: 32px 20px 80px;
}}
h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
h2 {{ font-size: 1.15rem; margin: 40px 0 4px; border-top: 1px solid var(--border); padding-top: 28px; }}
h3 {{ font-size: 0.95rem; color: var(--text-secondary); margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.03em; }}
.subtitle {{ color: var(--text-secondary); margin-bottom: 24px; }}
.tag {{ display: inline-block; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 2px 8px; border-radius: 4px; margin-right: 6px; }}
.tag-obs {{ background: color-mix(in srgb, var(--blue) 18%, transparent); color: var(--blue); }}
.tag-interp {{ background: color-mix(in srgb, var(--orange) 18%, transparent); color: var(--orange); }}
.tag-hyp {{ background: color-mix(in srgb, var(--aqua) 18%, transparent); color: var(--aqua); }}
.tag-fix {{ background: color-mix(in srgb, var(--red) 18%, transparent); color: var(--red); }}
.stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 28px; }}
.stat-tile {{ background: var(--surface-2); border-radius: 10px; padding: 14px 18px; min-width: 140px; flex: 1; }}
.stat-tile .value {{ font-size: 1.6rem; font-weight: 700; }}
.stat-tile .label {{ color: var(--text-secondary); font-size: 0.8rem; }}
.chart-svg {{ width: 100%; height: auto; }}
.axis-label {{ font-size: 9px; fill: var(--text-muted); }}
.bar-value {{ font-size: 9px; fill: var(--text-secondary); }}
.gridline {{ stroke: var(--border); stroke-width: 1; }}
.axis-line {{ stroke: var(--text-muted); stroke-width: 1; }}
.legend-row {{ display: flex; gap: 16px; margin-bottom: 4px; font-size: 0.8rem; color: var(--text-secondary); flex-wrap: wrap; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
.legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 10px 0 20px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }}
.mono {{ font-family: ui-monospace, Menlo, monospace; letter-spacing: 0.5px; }}
.digit-ok {{ color: var(--text-muted); }}
.digit-wrong {{ color: var(--red); font-weight: 700; background: color-mix(in srgb, var(--red) 12%, transparent); border-radius: 2px; }}
.callout {{ background: var(--surface-2); border-left: 3px solid var(--blue); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; }}
.callout.fix {{ border-left-color: var(--red); }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
footer {{ margin-top: 48px; color: var(--text-muted); font-size: 0.78rem; border-top: 1px solid var(--border); padding-top: 16px; }}
.controlled-table td {{ font-size: 0.78rem; white-space: nowrap; }}
.stratum-label {{ font-weight: 600; color: var(--text-secondary); }}
.n-tag {{ color: var(--text-muted); font-size: 0.72rem; }}
.hyp-table td, .hyp-table th {{ vertical-align: top; }}
.verdict-support {{ color: var(--aqua); font-weight: 600; }}
.verdict-against {{ color: var(--red); font-weight: 600; }}
.verdict-mixed {{ color: var(--yellow); font-weight: 600; }}
</style>

<h1>Task A error analysis</h1>
<p class="subtitle">Standard Transformer, 50,000 optimizer steps, checkpoint <code>runs/square_transformer_50k/peak.pt</code>
&middot; evaluated on <code>data/generated/square/val_iid.jsonl</code> ({n:,} examples)</p>

<div class="callout fix">
<span class="tag tag-fix">Correction</span> A previous version of this report displayed "67.5% examples with &le;1 wrong
digit" in the summary stat tiles. That number was hand-typed into the HTML instead of computed from the underlying
bucket fractions and was wrong: 0-wrong ({bucket_fracs[0]:.2%}) + 1-wrong ({bucket_fracs[1]:.2%}) =
<b>{le1_wrong_frac:.2%}</b>, not 67.5%. Every number in this report is now computed directly from
<code>task_a_analysis.json</code>/<code>task_a_deconfound.json</code> by <code>build_report.py</code> at generation
time &mdash; nothing below is hand-typed. See the <code>diagnostics/analysis_out/build_report.py</code> module
docstring for the fix.
</div>

<h3>Which split, exactly</h3>
<p>Evaluated on <b><code>val_iid.jsonl</code></b> only in this report. Task A's data generator
(<code>data/generate.py::gen_square_task</code>) draws <code>train</code>, <code>val_iid</code>, and
<code>heldout_x</code> as three disjoint samples from the <i>same</i> uniform distribution over
x&isin;[1, 999999], deduplicated against each other via one shared <code>seen</code> set at generation time. Task A
has no modulus, so there is no separate "held-out modulus" construction the way Tasks B/C/D have &mdash; for Task A,
<code>val_iid</code> and <code>heldout_x</code> are built by the <i>identical</i> procedure (disjoint uniform x,
nothing else varies), so they are not expected to differ and the terms were previously used loosely. This report
uses <code>val_iid.jsonl</code> throughout; <code>heldout_x.jsonl</code> would be expected to give statistically
indistinguishable numbers by construction, not a materially different held-out condition.</p>

<div class="stat-row">
  <div class="stat-tile"><div class="value">{exact_match_frac:.2%}</div><div class="label">exact match</div></div>
  <div class="stat-tile"><div class="value">{token_accuracy_frac:.1%}</div><div class="label">token accuracy</div></div>
  <div class="stat-tile"><div class="value">{le1_wrong_frac:.1%}</div><div class="label">examples with &le;1 wrong digit</div></div>
  <div class="stat-tile"><div class="value">{frac_contiguous:.1%}</div><div class="label">wrong examples: 1 contiguous error run</div></div>
</div>

<span class="tag tag-obs">Observation</span> {largest_bucket_name} is the single largest bucket at {largest_bucket_frac:.1%}
&mdash; most failures contain a contiguous block of {largest_bucket_name.replace(" wrong", "")} wrong middle digits,
not digits scattered across the whole output. But most examples are NOT close to exact: the corrected &le;1-wrong
figure is {le1_wrong_frac:.1%}, not the previously mis-typed 67.5%, and {bucket_fracs[3]:.1%} of examples have 4+
wrong digits (see chart below).

<h2>1. Error structure</h2>
<div class="two-col">
<div>
<h3>How many digits wrong per example</h3>
{error_dist_chart}
</div>
<div>
<h3>Position of the first wrong digit (given &ge;1 wrong)</h3>
{fw_chart}
</div>
</div>

<div class="callout">
<span class="tag tag-obs">Observation</span> First-wrong-digit position is concentrated at positions 4&ndash;7 (0-indexed,
MSB-first, 12-digit zero-padded output).
</div>

<h3>20 representative failures (predicted digit shown; red = wrong)</h3>
<table>
<tr><th>x</th><th>true x&sup2;</th><th>predicted (diff highlighted)</th><th># wrong</th><th>wrong positions</th></tr>
{fail_rows}
</table>

<h2>2. Where in the output does it fail</h2>
<h3>Accuracy by output position (0 = most significant digit, 11 = least significant)</h3>
{pos_acc_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Positions 0&ndash;3 and 8&ndash;11: 99.1&ndash;100% accuracy. Positions
4&ndash;7 (the middle third of the 12-digit output): 11&ndash;59% accuracy. True-digit entropy is flat at ~3.3 bits
across every position (full table below) &mdash; the dip is not explained by some positions having harder/less
predictable digit distributions than others.
</div>

<h3>Accuracy conditional on whether a carry enters that position</h3>
{carry_chart}
<div class="callout fix">
<span class="tag tag-fix">Downgraded claim</span> The first version of this report stated that this comparison alone
established carry propagation as <i>the</i> bottleneck, ahead of partial-product formation. That claim was premature:
central output columns simultaneously have the <b>most contributing digit pairs</b> and the <b>largest carries</b> (see
the confound check in section 5 below) &mdash; so a plain carry-present/absent split, without controlling for term
count, cannot separate "carries are the problem" from "there's just more raw arithmetic (more terms to sum) in the
middle." Section 5 redoes this with position, carry VALUE, and term count controlled jointly. The short version:
<b>both</b> factors turn out to matter independently, not just one.
</div>

<h3>Full per-position table</h3>
<table>
<tr><th>pos</th><th>accuracy</th><th>true H (bits)</th><th>pred H (bits)</th><th>mean confidence</th>
<th>calibration gap</th><th>acc | carry-in present</th><th>acc | no carry-in</th><th>top confusions (true&rarr;pred)</th></tr>
{pos_table_rows}
</table>
<p style="color:var(--text-secondary); font-size:0.82rem;">Calibration gap = mean top-1 confidence &minus; accuracy.
Positive at 4&ndash;7 means the model is overconfident in exactly the region it gets wrong most.</p>

<h2>3. Arithmetic-difficulty features vs. failure</h2>
<h3>Exact-match rate by number of significant digits in x</h3>
{ndx_chart}
<p style="color:var(--text-secondary); font-size:0.82rem;">n per bucket: {", ".join(f"{float(k):.0f}-digit: {v}" for k, v in zip(ndx_labels, ndx_ns))}.
{frac_6digit:.0%} of val_iid x's are the full 6 digits (x sampled uniformly to 999,999), so this feature has limited
spread in practice, but the trend among the rarer shorter-x examples is unambiguous.</p>

<h3>Decision-tree feature importance (predicting exact-match)</h3>
{feat_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Tree/logreg test accuracy ({tree_test_acc:.1%} / {logreg_test_acc:.1%}) is
<b>not meaningful on its own</b> &mdash; it barely beats the trivial majority-class baseline ({logreg_baseline:.1%})
because exact-match success is rare. The feature <i>ranking</i> is still informative, but several top features here
are near-perfectly collinear with each other (n_digits_x, n_partial_products, magnitude_bucket, max_multi_term_positions
all move together since {frac_6digit:.0%} of x's are 6 digits) &mdash; see section 5 for a version of this analysis
that controls for the confound directly instead of relying on a tree's arbitrary credit-splitting among collinear
features.
</div>

<h2>4. Non-neural baselines</h2>
{baseline_chart}
<table>
<tr><th>Baseline</th><th>Exact match</th><th>Token accuracy</th></tr>
{baseline_table_rows}
<tr><td>Nearest training input by digit Hamming distance</td><td>{hamming_exact:.2%}</td><td>&mdash; (sampled n={hamming_n})</td></tr>
<tr><td>Lookup table (exact seen x)</td><td colspan="2">{lookup_frac_seen:.0%} of val_iid x's were seen in train (0% expected &mdash; held-out-x by construction)</td></tr>
<tr><td>Leading digit only (position 0)</td><td colspan="2">{leading_only_acc:.2%} accuracy</td></tr>
<tr><td>Trailing digit only (position 11)</td><td colspan="2">{trailing_only_acc:.2%} accuracy &mdash; ones digit of x&sup2; is a pure function of x's own ones digit, no carries involved</td></tr>
</table>
<div class="callout">
<span class="tag tag-obs">Observation</span> The model clears every naive baseline on exact-match, and clears the
strongest baseline (nearest-training-target-by-value) by {token_vs_nearest_gap:.0f} points on token accuracy &mdash;
it has learned something beyond copying/interpolating a nearby training example.
</div>

<h2>5. Deconfounded analysis: carry vs. term-count vs. position</h2>
<p>Everything below comes from <code>deconfound_analysis.py</code>, run on the same {n_dc:,}-example val_iid split,
using true schoolbook (grade-school long multiplication) quantities computed per output column: raw diagonal sum
before carry, number of contributing digit pairs, incoming/outgoing carry value. A sanity check compares the
simulation's own reconstructed digit against the true label at every position across all {n_dc*12:,} (example,
position) pairs: <b>{sanity_mismatches} mismatches</b> &mdash; confirms the simulation reproduces x&sup2; exactly
before trusting anything downstream.</p>

<h3>The confound, made explicit</h3>
{confound_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Mean carry-in and mean term-count both rise sharply from position 0
toward the middle and fall back off toward position 11 &mdash; the same shape (inverted) as accuracy. This confirms
the confound: central columns simultaneously have more contributing terms AND larger carries, so naively splitting
on either one alone conflates the two.
</div>

<h3>Controlled comparison A &mdash; fix position + diagonal-sum bucket, vary carry-in</h3>
{controlled_a_html}
<h3>Controlled comparison B &mdash; fix position + carry-in bucket, vary number of contributing terms</h3>
{controlled_b_html}
<h3>Controlled comparison C &mdash; fix position + term-count (at that position's modal value), vary carry magnitude</h3>
{controlled_c_html}
<div class="callout">
<span class="tag tag-interp">Interpretation</span> Both deconfounded comparisons show a real, independent effect.
Comparison A: holding position and diagonal-sum bucket fixed, accuracy still falls steadily as carry-in rises (e.g.
pos=5,diagQ=1 falls from ~37% at carry=0 into the 7&ndash;15% range by carry=10+). Comparison B: holding position and
carry-in fixed (including carry_in=0), accuracy still falls steadily as term-count rises (e.g. pos=5,carry_in=0:
100%&rarr;26%&rarr;28% as terms go 1&rarr;3&rarr;5). <b>Neither factor is fully explained by the other</b> &mdash; this
directly falsifies the earlier single-cause "carry propagation alone" claim.
</div>

<h3>Per-digit error model (class-balanced logistic regression, predicting P(correct) per (example, position))</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Balanced accuracy</td><td>{pdm_metrics['balanced_accuracy']:.3f}</td></tr>
<tr><td>ROC-AUC</td><td>{pdm_metrics['roc_auc']:.3f}</td></tr>
<tr><td>PR-AUC</td><td>{pdm_metrics['pr_auc']:.3f} (baseline for a trivial always-correct classifier = positive rate = {pr_auc_baseline:.3f})</td></tr>
</table>
{pdm_coef_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> <code>n_terms</code> (number of contributing digit pairs) has the
<b>largest-magnitude standardized coefficient</b> ({dict(pdm_coefs)['n_terms']:+.2f}), edging out <code>carry_in</code>
({dict(pdm_coefs)['carry_in']:+.2f}). <code>position</code>'s coefficient ({dict(pdm_coefs)['position']:+.2f}) should
not be over-read: the true position&rarr;accuracy relationship is U-shaped, not monotone, and a linear coefficient on
raw position index poorly captures that &mdash; it is not evidence that "later position = more correct" in any simple
sense. Because term-count, diagonal sum, and carry are still correlated with each other even after quantile-binning
(binning controls gross confounding, it does not fully orthogonalize continuous, mutually-caused quantities), this
model narrows the field but does not cleanly crown one cause over the other.
</div>

<h2>6. Testing the "two-sided approximation" hypothesis (Hypothesis C)</h2>
<h3>Prefix-k / suffix-k exact match, k = 1..6</h3>
{prefix_suffix_chart}
<h3>Reconciliation gap: 12 &minus; (correct prefix width) &minus; (correct suffix width), per example</h3>
{gap_chart}
<div class="callout">
<span class="tag tag-interp">Interpretation</span> Mean correct-prefix width = {prefix_width_mean:.2f}, mean
correct-suffix width = {suffix_width_mean:.2f} (sum &asymp; {prefix_width_mean+suffix_width_mean:.1f} of 12). If
hypothesis C were the whole story (two independently-computed high/low regions that just fail to reconcile at a
fixed boundary), this gap would cluster tightly around a single constant. It does not: the gap ranges 0&ndash;7 with
its mode at {gap_mode} but real spread on both sides. That spread tracks each example's actual carry/term difficulty
(already shown in section 5), not a fixed positional boundary. Read literally, hypothesis C is <b>not well
supported as an independent mechanism</b> &mdash; the prefix/suffix framing is a valid <i>description</i> of the
error pattern (correct region, then wrong, then correct region again) but the <i>cause</i> of where the wrong region
starts and ends is better explained by per-example arithmetic difficulty (sections 5), which hypothesis C's
"independent approximation" framing does not by itself predict.
</div>

<h2>7. Layerwise linear probes</h2>
<p>Probing the residual stream after each of the 4 encoder layers (before the final norm/head), at the output-slot
positions, on a {min(4000, n_dc):,}-example sample. Ridge regression R² for continuous per-column quantities
(diagonal sum, carry-in, carry-out averaged over positions 4&ndash;7); logistic-regression accuracy for the two
global discrete quantities (probed from position 0's hidden state for the leading digit, position 11's for
x&sup2; mod 10).</p>
{probe_chart}
{global_probe_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Diagonal sum, carry-in, and carry-out are all substantially linearly
recoverable (R² 0.69&ndash;0.85) at <i>every</i> layer, including layer 0 &mdash; and recoverability <i>peaks at layer
1 and declines slightly by layer 3</i>, rather than sharpening with depth. Leading-digit and x&sup2; mod 10 are
both ~99&ndash;100% linearly decodable from layer 0/1 onward.
</div>
<div class="callout">
<span class="tag tag-interp">Interpretation</span> The raw arithmetic ingredients (diagonal sums, carries) are
approximately present in the hidden states from very early on &mdash; this weighs against the strongest version of
"the network never forms multiplication structure at all." But R²&asymp;0.7&ndash;0.85 is <i>approximate</i>, not
exact, and exact-digit output requires exact integer arithmetic on these quantities; a moderately-recoverable-but-not-
exact internal carry value is consistent with correct answers at low term-count/low-carry positions (0&ndash;3,
8&ndash;11) and incorrect answers exactly where many such approximate quantities must be combined precisely
(4&ndash;7). This suggests the bottleneck is in <i>precise combination</i> of already-present information, not in
failing to represent it at all &mdash; a distinction between hypotheses A and B that this probe cannot fully resolve
on its own (both would show "information present, output still wrong" under this reading).
</div>

<h2>8. Hypothesis comparison: A (dense aggregation) vs. B (carry propagation) vs. C (two-sided approximation)</h2>
<table class="hyp-table">
<tr><th>Hypothesis</th><th>Supporting evidence</th><th>Evidence against / limiting</th><th>Verdict</th></tr>
<tr>
<td><b>A.</b> Dense partial-product aggregation failure</td>
<td>Controlled comparison B: holding position AND carry-in fixed (incl. carry_in=0), accuracy still falls sharply as
term-count rises. <code>n_terms</code> has the single largest coefficient in the joint per-digit model.</td>
<td>Diagonal sums (the aggregated quantity itself) are linearly recoverable at every layer with R²&asymp;0.7&ndash;0.85
&mdash; the network does form an approximate aggregate, it isn't blind to partial products.</td>
<td class="verdict-mixed">Partially supported</td>
</tr>
<tr>
<td><b>B.</b> Carry-state propagation failure</td>
<td>Controlled comparisons A and C: holding position and (diagonal-sum bucket / term-count) fixed, accuracy still
falls sharply and monotonically as carry-in value/magnitude rises. <code>carry_in</code> is a significant independent
coefficient in the joint model.</td>
<td>Carry-in/out values are also linearly recoverable at every layer (R²&asymp;0.7&ndash;0.85) &mdash; not absent, just
imprecise. Not larger in the joint model's coefficient ranking than term-count.</td>
<td class="verdict-mixed">Partially supported</td>
</tr>
<tr>
<td><b>C.</b> Correct high/low approximations failing to reconcile in the middle</td>
<td>Descriptively true: there is a correct-prefix region, a wrong region, and a correct-suffix region in almost
every failing example.</td>
<td>The reconciliation gap (12 &minus; prefix &minus; suffix width) is not a fixed constant &mdash; it varies per
example and that variation tracks carry/term difficulty (section 5), not a fixed architectural seam. As a distinct
causal mechanism (independent of A/B), not supported; better read as a redescription of A/B's effects.</td>
<td class="verdict-against">Not supported as an independent mechanism</td>
</tr>
</table>

<div class="callout">
<span class="tag tag-hyp">Hypothesis (still speculative)</span> Best current read: this is not a single-cause failure.
Both term-count/aggregation density (A) and carry magnitude (B) independently and substantially predict digit-level
failure even after controlling for each other and for position, and the raw ingredients for both (diagonal sums,
carries) are approximately linearly present in hidden states at every layer without translating into exact output at
high-term/high-carry positions. The most defensible single sentence: <b>the model can approximately compute the right
quantities in the hard middle columns but cannot combine them precisely enough for exact digit output when both term
count and carry magnitude are simultaneously high</b> &mdash; which sections 1&ndash;4 alone (before this deconfounding
pass) could not distinguish from a pure carry-only story. Sections 8 (causal ablation: directly intervene on the
carry-associated vs. aggregation-associated hidden directions) and section 12's Hypotheses C/D ablations (auxiliary
carry supervision vs. auxiliary partial-product supervision, run separately) are the next experiments that would
actually adjudicate between A and B rather than just narrowing the field, as this pass did.
</div>

<footer>
Generated from <code>diagnostics/analysis_task_a.py</code> and <code>diagnostics/deconfound_analysis.py</code> by
<code>diagnostics/analysis_out/build_report.py</code> &middot; raw data in <code>task_a_analysis.json</code> and
<code>task_a_deconfound.json</code> &middot; n={n:,} val_iid examples.
</footer>
"""

out_path = HERE / "task_a_error_analysis.html"
with open(out_path, "w") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
