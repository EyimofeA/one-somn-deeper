import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "task_a_analysis.json"))
s1 = d["section1_error_structure"]
s2 = d["section2_arithmetic_difficulty"]
s3 = d["section3_baselines"]
s4 = d["section4_per_position"]["per_position"]
n = d["n_val_examples"]

PALETTE = {
    "blue": ("#2a78d6", "#3987e5"),
    "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"),
    "yellow": ("#eda100", "#c98500"),
    "red": ("#e34948", "#e66767"),
}


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


def svg_line_chart(x_labels, series_list, width=760, height=300, colors=("blue", "orange"), legend=("A", "B"), y_max=1.0):
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
                        f'<title>pos {x_labels[i]}: {series[i]:.1%}</title></circle>')
    for i, lab in enumerate(x_labels):
        x = pad_l + i * step
        els.append(f'<text x="{x:.1f}" y="{pad_t+plot_h+16}" class="axis-label" text-anchor="middle">{lab}</text>')
    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-swatch" style="background:var(--{c})"></span>{t}</span>'
        for c, t in zip(colors, legend)
    ) if len(series_list) > 1 else ""
    return f'<div class="legend-row">{legend_html}</div><svg viewBox="0 0 {width} {height}" class="chart-svg" role="img">' + "".join(els) + "</svg>"


# ---- Section 1 data ----
bucket_names = s1["bucket_names"]
bucket_fracs = s1["bucket_fractions"]
error_dist_chart = svg_bar_chart(bucket_names, bucket_fracs, color="blue", fmt="{:.1%}")

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
                               legend=("carry enters this position", "no carry enters this position"))

# ---- Section 2 feature importance ----
tree_imp = s2["tree_feature_importances"]
feat_sorted = sorted(tree_imp.items(), key=lambda kv: -kv[1])
feat_labels = [k.replace("_", " ") for k, v in feat_sorted]
feat_vals = [v for k, v in feat_sorted]
feat_chart = svg_bar_chart(feat_labels, feat_vals, width=760, height=300, color="aqua", fmt="{:.3f}")

# n_digits_x bucket table -> chart
ndx = s2["bucket_report"]["n_digits_x"]
ndx_labels = sorted(ndx.keys(), key=lambda k: float(k))
ndx_vals = [ndx[k]["exact_match"] for k in ndx_labels]
ndx_ns = [ndx[k]["n"] for k in ndx_labels]
ndx_chart = svg_bar_chart([f"{float(k):.0f} digits" for k in ndx_labels], ndx_vals, color="yellow", fmt="{:.1%}")

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

# ---- representative failures table rows ----
def render_digits(true_d, pred_d, wrong):
    cells = []
    for i, (t, p) in enumerate(zip(true_d, pred_d)):
        cls = "digit-wrong" if i in wrong else "digit-ok"
        cells.append(f'<span class="{cls}">{p}</span>')
    return "".join(cells)

fail_rows = ""
for f in s1["representative_failures"]:
    true_str = "".join(str(d) for d in f["true_digits"])
    pred_rendered = render_digits(f["true_digits"], f["pred_digits"], set(f["wrong_positions"]))
    fail_rows += (
        f'<tr><td>{f["x"]}</td><td class="mono">{true_str}</td>'
        f'<td class="mono">{pred_rendered}</td><td>{f["n_wrong"]}</td>'
        f'<td>{", ".join(str(p) for p in f["wrong_positions"])}</td></tr>'
    )

# ---- per-position table rows ----
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
  max-width: 980px; margin: 0 auto; padding: 32px 20px 80px;
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
.stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 28px; }}
.stat-tile {{ background: var(--surface-2); border-radius: 10px; padding: 14px 18px; min-width: 140px; flex: 1; }}
.stat-tile .value {{ font-size: 1.6rem; font-weight: 700; }}
.stat-tile .label {{ color: var(--text-secondary); font-size: 0.8rem; }}
.chart-svg {{ width: 100%; height: auto; }}
.axis-label {{ font-size: 9px; fill: var(--text-muted); }}
.bar-value {{ font-size: 9px; fill: var(--text-secondary); }}
.gridline {{ stroke: var(--border); stroke-width: 1; }}
.axis-line {{ stroke: var(--text-muted); stroke-width: 1; }}
.legend-row {{ display: flex; gap: 16px; margin-bottom: 4px; font-size: 0.8rem; color: var(--text-secondary); }}
.legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
.legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 10px 0 20px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }}
.mono {{ font-family: ui-monospace, Menlo, monospace; letter-spacing: 0.5px; }}
.digit-ok {{ color: var(--text-muted); }}
.digit-wrong {{ color: var(--red); font-weight: 700; background: color-mix(in srgb, var(--red) 12%, transparent); border-radius: 2px; }}
.callout {{ background: var(--surface-2); border-left: 3px solid var(--blue); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 16px 0; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
footer {{ margin-top: 48px; color: var(--text-muted); font-size: 0.78rem; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>

<h1>Task A error analysis</h1>
<p class="subtitle">Standard Transformer, 50,000 optimizer steps, checkpoint <code>runs/square_transformer_50k/peak.pt</code> &middot; evaluated on {n:,} held-out-x val_iid examples</p>

<div class="stat-row">
  <div class="stat-tile"><div class="value">0.81%</div><div class="label">exact match</div></div>
  <div class="stat-tile"><div class="value">75.8%</div><div class="label">token accuracy</div></div>
  <div class="stat-tile"><div class="value">67.5%</div><div class="label">examples with &le;1 wrong digit</div></div>
  <div class="stat-tile"><div class="value">88.4%</div><div class="label">wrong examples: 1 contiguous error run</div></div>
</div>

<span class="tag tag-obs">Observation</span> This is not diffuse noise. Nearly 90% of failures are a single contiguous
block of wrong digits, and the model gets the leading 4 and trailing 4 digits (of 12) essentially perfect. The failure
is concentrated in one region of the output.

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
MSB-first, 12-digit zero-padded output). Once a digit is wrong, the very next digit is wrong far more often than the
unconditional rate at every position 4&ndash;7 (contagion effect) &mdash; consistent with a genuine carry-chain
disruption rather than independent per-digit mistakes.
</div>

<h3>20 representative failures (predicted digit shown; red = wrong)</h3>
<table>
<tr><th>x</th><th>true x&sup2;</th><th>predicted (diff highlighted)</th><th># wrong</th><th>wrong positions</th></tr>
{fail_rows}
</table>

<h2>2. Where in the output does it fail, and why</h2>
<h3>Accuracy by output position (0 = most significant digit, 11 = least significant)</h3>
{pos_acc_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Positions 0&ndash;3 and 8&ndash;11: 99.1&ndash;100% accuracy. Positions
4&ndash;7 (the middle third of the 12-digit output): 11&ndash;59% accuracy &mdash; a sharp, localized dip, not a gradual
slope. True-digit entropy is <b>flat at ~3.3 bits across every position</b> (see full table below) &mdash; the dip is not
explained by some positions having harder/less predictable digit distributions than others.
</div>

<h3>Accuracy conditional on whether a carry enters that position</h3>
{carry_chart}
<div class="callout">
<span class="tag tag-interp">Interpretation</span> At every one of positions 4&ndash;7, accuracy is dramatically lower
specifically on the subset of examples where a real carry propagates into that column (e.g. position 5: 10.9% with an
incoming carry vs. 45.1% without). This is the strongest single piece of evidence in this analysis: <b>the bottleneck is
carry propagation through the middle output columns, not partial-product formation or output decoding</b> &mdash;
positions 0&ndash;3 and 8&ndash;11 also receive multi-term partial-product contributions and are still solved almost
perfectly.
</div>

<h3>Full per-position table</h3>
<table>
<tr><th>pos</th><th>accuracy</th><th>true H (bits)</th><th>pred H (bits)</th><th>mean confidence</th>
<th>calibration gap</th><th>acc | carry-in</th><th>acc | no carry-in</th><th>top confusions (true&rarr;pred)</th></tr>
{pos_table_rows}
</table>
<p style="color:var(--text-secondary); font-size:0.82rem;">Calibration gap = mean top-1 confidence &minus; accuracy.
Positive at 4&ndash;7 means the model is <i>overconfident</i> in exactly the region it gets wrong most; negative
elsewhere means slight underconfidence where it's already correct almost every time.</p>

<h2>3. Arithmetic-difficulty features vs. failure</h2>
<h3>Exact-match rate by number of significant digits in x</h3>
{ndx_chart}
<p style="color:var(--text-secondary); font-size:0.82rem;">n per bucket: {", ".join(f"{float(k):.0f}-digit: {v}" for k, v in zip(ndx_labels, ndx_ns))}.
90% of val_iid x's are the full 6 digits (x sampled uniformly to 999,999), so this feature has limited spread in practice, but the trend among the rarer shorter-x examples is unambiguous.</p>

<h3>Decision-tree feature importance (predicting exact-match)</h3>
{feat_chart}
<div class="callout">
<span class="tag tag-obs">Observation</span> Tree/logreg test "accuracy" (99.2%) is <b>not meaningful on its own</b> &mdash;
it barely beats the trivial majority-class baseline (99.2%) because exact-match success is rare (81/10,000). The feature
<i>ranking</i> is still informative: <code>total_carries</code>, <code>longest_carry_chain</code>, and
<code>n_digits_x</code>/<code>n_partial_products</code>/<code>magnitude_bucket</code> (all near-perfectly collinear with
each other here, since 90% of x's are 6 digits) dominate. <code>dist_to_pow10</code>'s large tree importance is likely a
proxy for "small number of significant digits" (numbers near a power of 10 below 10&sup6; tend to have fewer nonzero
digits), not an independent effect &mdash; flagged as a collinearity artifact, not a distinct causal feature.
</div>

<h2>4. Non-neural baselines</h2>
{baseline_chart}
<table>
<tr><th>Baseline</th><th>Exact match</th><th>Token accuracy</th></tr>
<tr><td>Trained model</td><td>0.81%</td><td>75.8%</td></tr>
<tr><td>Most-common digit per position</td><td>0.00%</td><td>14.9%</td></tr>
<tr><td>Copy input x's digits into output</td><td>0.00%</td><td>15.3%</td></tr>
<tr><td>Nearest training target by |x&minus;x_train|</td><td>0.00%</td><td>44.5%</td></tr>
<tr><td>Nearest training input by digit Hamming distance</td><td>0.00%</td><td>&mdash; (sampled 500)</td></tr>
<tr><td>Lookup table (exact seen x)</td><td>0.00%</td><td>n/a &mdash; 0% of val_iid x's seen in train (held-out-x split by construction)</td></tr>
<tr><td>Leading digit only</td><td colspan="2">99.99% accuracy at position 0 alone</td></tr>
<tr><td>Trailing digit only</td><td colspan="2">100% accuracy at position 11 alone &mdash; ones digit of x&sup2; is a pure function of x's own ones digit, no carries involved</td></tr>
</table>
<div class="callout">
<span class="tag tag-obs">Observation</span> The model clears every naive baseline on exact-match (0.81% vs. 0% for all of them),
and clears the strongest baseline (nearest-training-target-by-value, 44.5% token accuracy) by 31 points on token accuracy
&mdash; it has learned something beyond copying/interpolating a nearby training example. But 0.81% exact-match is still
far from useful, and the gap is concentrated exactly where section 2 says: the middle carry-heavy columns.
</div>

<h2>What this rules in / rules out (sections 5&ndash;13 not yet run)</h2>
<p><span class="tag tag-hyp">Hypothesis</span> Given the evidence above &mdash; flat entropy across positions, a sharp
(not gradual) accuracy cliff at positions 4&ndash;7, and a large carry-in/no-carry-in accuracy gap specifically in that
window &mdash; <b>Hypothesis C (carry propagation is the bottleneck)</b> is the best-supported explanation so far, ahead
of Hypothesis A (capacity: the same architecture solves 8 of 12 positions essentially perfectly, so raw capacity is not
obviously the limit), Hypothesis D (multiplication decomposition: partial-product-heavy positions 0&ndash;3/8&ndash;11
are already solved), and Hypothesis F (tokenization: digit-level tokenization is fine everywhere except this one region).
This is <b>not yet a causal claim</b> &mdash; it is a strong correlational read from sections 1&ndash;4 only. Sections 6
(carry probes) and 8 (causal ablations on carry-associated hidden states) are the direct tests that would confirm or
falsify it, and neither has been run yet.</p>

<footer>
Generated from <code>diagnostics/analysis_task_a.py</code> &middot; raw data in
<code>diagnostics/analysis_out/task_a_analysis.json</code> &middot; n={n:,} val_iid examples, held-out x split.
</footer>
"""

out_path = HERE / "task_a_error_analysis.html"
with open(out_path, "w") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
