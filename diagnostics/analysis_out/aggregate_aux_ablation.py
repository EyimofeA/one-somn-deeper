"""Aggregates the 15 aux-ablation runs (5 conditions x 3 seeds) into one JSON
summary: per-condition mean/std of exact match, per-position accuracy,
accuracy-by-carry-bucket, param counts, throughput, and full convergence
curves for plotting. Run after train_aux_ablation.py's 15 runs complete.
"""

import glob
import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE.parent / "runs" / "aux_ablation"
CONDITIONS = ["baseline", "carry", "diagonal", "both", "both_annealed"]
SEEDS = [0, 1, 2]


def mean_std(vals):
    vals = list(vals)
    m = stats.mean(vals)
    s = stats.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": m, "std": s, "values": vals}


def main():
    per_run = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            run_dir = RUNS_DIR / f"{cond}_seed{seed}"
            report = json.loads((run_dir / "eval_report.json").read_text())
            curve = [json.loads(l) for l in (run_dir / "learning_curve.jsonl").open()]
            per_run[f"{cond}_seed{seed}"] = {"report": report, "curve": curve}

    summary = {"conditions": {}}
    for cond in CONDITIONS:
        runs = [per_run[f"{cond}_seed{s}"] for s in SEEDS]
        reports = [r["report"] for r in runs]
        curves = [r["curve"] for r in runs]

        best_exact = mean_std(r["best_val_exact_match"] for r in reports)
        final_exact = mean_std(r["final_val_metrics"]["exact_match"] for r in reports)
        final_token = mean_std(r["final_val_metrics"]["token_accuracy"] for r in reports)
        train_exact = mean_std(c[-1]["train_exact_match"] for c in curves)
        steps_per_sec = mean_std(stats.mean(row["steps_per_sec"] for row in c) for c in curves)

        # per-position accuracy, averaged across seeds
        n_pos = 12
        per_position = {}
        for p in range(n_pos):
            vals = [r["post_hoc_analysis"]["per_position_accuracy"][str(p)] for r in reports]
            per_position[p] = mean_std(vals)

        # accuracy by carry-in bucket, averaged across seeds
        carry_buckets = list(reports[0]["post_hoc_analysis"]["accuracy_by_carry_in_bucket"].keys())
        by_carry = {}
        for cb in carry_buckets:
            vals = [r["post_hoc_analysis"]["accuracy_by_carry_in_bucket"][cb]["accuracy"] for r in reports if cb in r["post_hoc_analysis"]["accuracy_by_carry_in_bucket"]]
            if vals:
                by_carry[cb] = mean_std(vals)

        run_cfg = reports[0]["run_config"]
        convergence = [
            {"step": row["step"], "val_exact_match": row["val_exact_match"], "train_exact_match": row["train_exact_match"],
             "aux_carry_weight": row.get("aux_carry_weight"), "aux_diag_weight": row.get("aux_diag_weight"),
             "train_aux_carry_mse": row.get("train_aux_carry_mse"), "train_aux_diag_mse": row.get("train_aux_diag_mse")}
            for row in curves[0]  # representative seed-0 curve for plotting
        ]
        # all-seed convergence (for shaded band / all lines)
        convergence_all_seeds = [
            [{"step": row["step"], "val_exact_match": row["val_exact_match"]} for row in c] for c in curves
        ]

        summary["conditions"][cond] = {
            "n_params_total": run_cfg["n_params_total"],
            "n_params_backbone": run_cfg["n_params_backbone"],
            "n_params_aux_heads": run_cfg["n_params_aux_heads"],
            "use_carry_aux": run_cfg["use_carry_aux"],
            "use_diag_aux": run_cfg["use_diag_aux"],
            "anneal": run_cfg["anneal"],
            "best_val_exact_match": best_exact,
            "final_val_exact_match": final_exact,
            "final_val_token_accuracy": final_token,
            "final_train_exact_match": train_exact,
            "steps_per_sec": steps_per_sec,
            "per_position_accuracy": per_position,
            "accuracy_by_carry_in_bucket": by_carry,
            "convergence_seed0": convergence,
            "convergence_all_seeds": convergence_all_seeds,
        }

    out_path = HERE / "aux_ablation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")

    # print a quick table
    print(f"\n{'condition':15s} {'best_exact':>12s} {'final_exact':>12s} {'final_token':>12s} {'train_exact':>12s} {'params':>8s} {'steps/s':>8s}")
    for cond in CONDITIONS:
        c = summary["conditions"][cond]
        print(f"{cond:15s} {c['best_val_exact_match']['mean']:.4f}±{c['best_val_exact_match']['std']:.3f}  "
              f"{c['final_val_exact_match']['mean']:.4f}±{c['final_val_exact_match']['std']:.3f}  "
              f"{c['final_val_token_accuracy']['mean']:.4f}±{c['final_val_token_accuracy']['std']:.3f}  "
              f"{c['final_train_exact_match']['mean']:.4f}±{c['final_train_exact_match']['std']:.3f}  "
              f"{c['n_params_total']:>8d} {c['steps_per_sec']['mean']:>7.1f}")


if __name__ == "__main__":
    main()
