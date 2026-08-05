"""B3: Task B (mod) error analysis by arithmetic difficulty, not just
aggregate accuracy. Mirrors analysis_task_a.py's structure/rigor: a
schoolbook-long-division feature simulation with a sanity check against the
true label, per-position accuracy/confidence/calibration, and non-neural
baselines -- run once a trained runs/mod_transformer_50k checkpoint exists
(B2). Correlational only -- do not read causal claims into feature-bucket
accuracy alone (see B4's competing-hypotheses gate).

Writes analysis_out/task_b_analysis.json.
"""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import DiagnosticDataset, load_jsonl
from data.tokens import NUM_MOD_DIGITS
from train import build_model

W = NUM_MOD_DIGITS  # 4


# ---------------------------------------------------------------------------
# reference reduction procedure: schoolbook long division, MSB-first, one
# digit of u brought down at a time -- the direct analogue of Task A's
# carry_features long-multiplication simulation.
# ---------------------------------------------------------------------------
def reduction_features(n: int, u: int) -> dict:
    u_digits = [int(c) for c in str(u)]  # MSB-first
    remainder = 0
    quotient_digits = []
    total_subtractions = 0  # how many times n was subtracted off across the whole process
    max_step_subtractions = 0  # largest single quotient digit produced in one step (0-9)
    for d in u_digits:
        remainder = remainder * 10 + d
        qd = 0
        while remainder >= n:
            remainder -= n
            qd += 1
        quotient_digits.append(qd)
        total_subtractions += qd
        max_step_subtractions = max(max_step_subtractions, qd)

    true_remainder = remainder  # should equal u % n exactly
    quotient = int("".join(str(d) for d in quotient_digits)) if quotient_digits else 0
    n_digits_u = len(str(u))
    n_digits_n = len(str(n))
    return {
        "quotient": quotient,
        "n_quotient_digits": len(str(quotient)),
        "n_nonzero_quotient_digits": sum(1 for d in quotient_digits if d != 0),
        "ratio_u_over_n": u / n,
        "n_digits_u": n_digits_u,
        "n_digits_n": n_digits_n,
        "remainder": true_remainder,
        "n_digits_remainder": len(str(true_remainder)),
        "dist_remainder_to_0": true_remainder,
        "dist_remainder_to_n": n - true_remainder,
        "u_less_than_n": u < n,
        "total_subtractions": total_subtractions,
        "max_step_subtractions": max_step_subtractions,
    }


def quotient_bucket(q: int) -> str:
    if q == 0:
        return "0"
    if q == 1:
        return "1"
    if q < 10:
        return "small(2-9)"
    return "large(10+)"


def ratio_bucket(ratio: float) -> str:
    if ratio < 1:
        return "<1"
    if ratio < 2:
        return "1-<2"
    if ratio < 10:
        return "2-<10"
    if ratio < 100:
        return "10-<100"
    return "100+"


# ---------------------------------------------------------------------------
# model loading + inference
# ---------------------------------------------------------------------------
def load_trained_model(run_dir: Path, checkpoint: str):
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    train_ds = DiagnosticDataset(cfg["data"]["train"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, max_seq_len=train_ds.max_len, task="mod").to(device)
    ckpt = run_dir / f"{checkpoint}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, cfg, str(ckpt), device


@torch.no_grad()
def run_inference(model, rows: list[dict], device: str, batch_size: int = 256):
    all_preds, all_probs = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        ids = torch.tensor([r["input_ids"] for r in chunk], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        logits = model(ids, mask)[:, -W:, :]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_probs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/mod_transformer_50k")
    ap.add_argument("--data-file", default=None, help="override config validation split")
    ap.add_argument("--checkpoint", choices=("peak", "final"), default="peak")
    ap.add_argument("--out-dir", default="analysis_out")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ckpt_path, device = load_trained_model(run_dir, args.checkpoint)
    data_file = Path(args.data_file) if args.data_file else Path(cfg["data"]["val"])
    rows = load_jsonl(data_file)
    print(f"loaded {len(rows)} rows from {data_file}, model={ckpt_path}, device={device}")

    preds, probs = run_inference(model, rows, device)
    targets = np.array([r["labels"][-W:] for r in rows])
    correct = preds == targets
    n = len(rows)

    print("Computing reference long-division features + sanity check...")
    feats = [reduction_features(int(r["n"]), int(r["u"])) for r in rows]
    true_remainders = np.array([r["y"] if "y" in r else int(r["u"]) % int(r["n"]) for r in rows])
    sim_remainders = np.array([f["remainder"] for f in feats])
    mismatches = int((sim_remainders != true_remainders).sum())
    print(f"sanity check: long-division simulation vs true u%n mismatches = {mismatches}/{n} (must be 0)")
    assert mismatches == 0, "reduction_features' simulated remainder does not match true u % n -- bug in feature extraction"

    # ---------------- error structure ----------------
    n_wrong = (~correct).sum(axis=1)
    bucket_counts = {
        "0 wrong": int((n_wrong == 0).sum()), "1 wrong": int((n_wrong == 1).sum()),
        "2-3 wrong": int(((n_wrong >= 2) & (n_wrong <= 3)).sum()), "4+ wrong": int((n_wrong == 4).sum()),
    }
    first_wrong_pos = [int(np.where(~correct[i])[0][0]) if n_wrong[i] > 0 else -1 for i in range(n)]
    first_wrong_hist = Counter(p for p in first_wrong_pos if p >= 0)

    wrong_idx = np.where(n_wrong > 0)[0]

    def longest_error_run(i: int) -> int:
        longest = current = 0
        for wrong in ~correct[i]:
            current = current + 1 if wrong else 0
            longest = max(longest, current)
        return longest

    error_run_lengths = [longest_error_run(i) for i in wrong_idx]
    contiguous = sum(run == n_wrong[i] for run, i in zip(error_run_lengths, wrong_idx))
    frac_contiguous = contiguous / len(wrong_idx) if len(wrong_idx) else None

    print("\n=== Error structure ===")
    print(f"exact match: {bucket_counts['0 wrong']}/{n} = {bucket_counts['0 wrong']/n:.4f}")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v} ({v/n:.4f})")
    print(f"first-wrong-position histogram: {dict(sorted(first_wrong_hist.items()))}")
    print(f"fraction of wrong examples with contiguous wrong positions: {frac_contiguous}")
    print(f"longest error-run histogram: {dict(sorted(Counter(error_run_lengths).items()))}")

    # ---------------- per-position accuracy/confidence/calibration ----------------
    print("\n=== Per-position accuracy/confidence/calibration ===")
    per_position = []
    for p in range(W):
        acc = float(correct[:, p].mean())
        conf = probs[:, p, :].max(axis=1)
        calib_gap = float(conf.mean() - acc)
        wrong_mask = ~correct[:, p]
        confusions = Counter(zip(targets[wrong_mask, p].tolist(), preds[wrong_mask, p].tolist()))
        per_position.append({
            "position": p, "accuracy": acc, "mean_confidence": float(conf.mean()),
            "calibration_gap": calib_gap, "top_confusions": confusions.most_common(3),
        })
        print(f"  pos {p}: acc={acc:.4f} conf={conf.mean():.3f} calib_gap={calib_gap:+.4f}")

    # ---------------- accuracy by arithmetic-difficulty feature ----------------
    print("\n=== Accuracy by arithmetic-difficulty feature ===")
    y_exact = (n_wrong == 0).astype(int)
    feature_bucket_report = {}

    def bucket_report(key_fn, name):
        buckets: dict[str, list[int]] = {}
        for i, f in enumerate(feats):
            k = key_fn(f)
            buckets.setdefault(str(k), []).append(i)
        rep = {k: {"n": len(idxs), "exact_match": float(y_exact[idxs].mean())} for k, idxs in buckets.items()}
        feature_bucket_report[name] = rep
        print(f"  {name}: " + ", ".join(f"{k}:{v['exact_match']:.3f}(n={v['n']})" for k, v in sorted(rep.items())))

    bucket_report(lambda f: quotient_bucket(f["quotient"]), "quotient_bucket")
    feature_bucket_report["quotient"] = {
        str(q): {"n": int(sum(f["quotient"] == q for f in feats)), "exact_match": float(y_exact[[f["quotient"] == q for f in feats]].mean())}
        for q in sorted({f["quotient"] for f in feats})
    }
    bucket_report(lambda f: len(str(f["quotient"])), "quotient_digit_length")
    bucket_report(lambda f: ratio_bucket(f["ratio_u_over_n"]), "u_over_n_ratio")
    bucket_report(lambda f: f["u_less_than_n"], "u_less_than_n")
    bucket_report(lambda f: f["n_digits_u"], "n_digits_u")
    bucket_report(lambda f: f["n_digits_n"], "n_digits_n")
    bucket_report(lambda f: min(f["n_digits_remainder"], 4), "n_digits_remainder")
    feature_bucket_report["remainder_value"] = {
        str(r): {"n": int(sum(f["remainder"] == r for f in feats)), "exact_match": float(y_exact[[f["remainder"] == r for f in feats]].mean())}
        for r in sorted({f["remainder"] for f in feats})
    }
    dist0 = np.array([f["dist_remainder_to_0"] for f in feats])
    distn = np.array([f["dist_remainder_to_n"] for f in feats])
    distnear = np.minimum(dist0, distn)
    dist0_q = np.digitize(dist0, np.quantile(dist0, [0.25, 0.5, 0.75]))
    distn_q = np.digitize(distn, np.quantile(distn, [0.25, 0.5, 0.75]))
    feature_bucket_report["dist_remainder_to_0_quartile"] = {
        str(q): {"n": int((dist0_q == q).sum()), "exact_match": float(y_exact[dist0_q == q].mean())}
        for q in np.unique(dist0_q)
    }
    feature_bucket_report["dist_remainder_to_n_quartile"] = {
        str(q): {"n": int((distn_q == q).sum()), "exact_match": float(y_exact[distn_q == q].mean())}
        for q in np.unique(distn_q)
    }
    distnear_q = np.digitize(distnear, np.quantile(distnear, [0.25, 0.5, 0.75]))
    feature_bucket_report["dist_to_nearest_multiple_quartile"] = {
        str(q): {"n": int((distnear_q == q).sum()), "exact_match": float(y_exact[distnear_q == q].mean())}
        for q in np.unique(distnear_q)
    }
    print("  dist_remainder_to_0_quartile: " + ", ".join(f"{k}:{v['exact_match']:.3f}(n={v['n']})" for k, v in sorted(feature_bucket_report["dist_remainder_to_0_quartile"].items())))
    print("  dist_remainder_to_n_quartile: " + ", ".join(f"{k}:{v['exact_match']:.3f}(n={v['n']})" for k, v in sorted(feature_bucket_report["dist_remainder_to_n_quartile"].items())))
    print("  dist_to_nearest_multiple_quartile: " + ", ".join(f"{k}:{v['exact_match']:.3f}(n={v['n']})" for k, v in sorted(feature_bucket_report["dist_to_nearest_multiple_quartile"].items())))

    # ---------------- baselines ----------------
    print("\n=== Non-neural baselines ===")
    train_rows = load_jsonl(cfg["data"]["train"])
    train_targets = np.array([r["labels"][-W:] for r in train_rows])
    train_n_arr = np.array([r["n"] for r in train_rows])
    train_u_arr = np.array([r["u"] for r in train_rows])

    mode_digit_per_pos = np.array([Counter(train_targets[:, p]).most_common(1)[0][0] for p in range(W)])
    baseline_mode = mode_digit_per_pos[None, :] == targets
    exact_mode, token_mode = float(baseline_mode.all(axis=1).mean()), float(baseline_mode.mean())

    zero_pred = np.zeros((n, W), dtype=int)
    baseline_zero = zero_pred == targets
    exact_zero, token_zero = float(baseline_zero.all(axis=1).mean()), float(baseline_zero.mean())

    # "output u unchanged": does u's own last 4 digits happen to equal the answer
    u_last4 = np.array([[int(c) for c in f"{int(r['u']):04d}"[-4:]] for r in rows])
    baseline_u_unchanged = u_last4 == targets
    exact_u_unchanged, token_u_unchanged = float(baseline_u_unchanged.all(axis=1).mean()), float(baseline_u_unchanged.mean())
    mod10_baselines, mod10_digits = {}, {}
    for width in (1, 2, 3):
        values = np.array([int(r["u"]) % (10 ** width) for r in rows])
        digits = np.array([[int(c) for c in f"{v:04d}"] for v in values])
        matches = digits == targets
        name = f"output_u_mod_10^{width}"
        mod10_digits[name] = digits
        mod10_baselines[name] = {"exact_match": float(matches.all(axis=1).mean()), "token_accuracy": float(matches.mean())}

    # nearest training example by u (numeric distance), read off its target directly
    order = np.argsort(train_u_arr)
    sorted_u = train_u_arr[order]
    u_arr = np.array([r["u"] for r in rows])
    idx = np.clip(np.searchsorted(sorted_u, u_arr), 0, len(sorted_u) - 1)
    left = np.clip(idx - 1, 0, len(sorted_u) - 1)
    use_left = np.abs(sorted_u[left] - u_arr) < np.abs(sorted_u[idx] - u_arr)
    chosen = np.where(use_left, left, idx)
    nearest_orig = order[chosen]
    baseline_nearest_u = train_targets[nearest_orig] == targets
    exact_nearest_u, token_nearest_u = float(baseline_nearest_u.all(axis=1).mean()), float(baseline_nearest_u.mean())

    # naive "nearest multiple" heuristic: predict u - n*round(u/n) using float division (breaks near exact multiples/negative wraps)
    naive_remainder = []
    for r in rows:
        nn, uu = int(r["n"]), int(r["u"])
        guess = uu - nn * round(uu / nn)
        naive_remainder.append(max(0, min(nn - 1, guess)))
    naive_digits = np.array([[int(c) for c in f"{v:04d}"] for v in naive_remainder])
    baseline_naive = naive_digits == targets
    exact_naive, token_naive = float(baseline_naive.all(axis=1).mean()), float(baseline_naive.mean())

    baselines = {
        "trained_model": {"exact_match": float(y_exact.mean()), "token_accuracy": float(correct.mean())},
        "most_common_digit_per_position": {"exact_match": exact_mode, "token_accuracy": token_mode},
        "output_zero": {"exact_match": exact_zero, "token_accuracy": token_zero},
        "output_u_unchanged": {"exact_match": exact_u_unchanged, "token_accuracy": token_u_unchanged},
        **mod10_baselines,
        "nearest_training_by_u": {"exact_match": exact_nearest_u, "token_accuracy": token_nearest_u},
        "naive_nearest_multiple_heuristic": {"exact_match": exact_naive, "token_accuracy": token_naive},
    }
    baseline_predictions = {
        "most_common_digit_per_position": np.broadcast_to(mode_digit_per_pos, targets.shape),
        "output_zero": zero_pred,
        "output_u_unchanged": u_last4,
        "nearest_training_by_u": train_targets[nearest_orig],
        "naive_nearest_multiple_heuristic": naive_digits,
        **mod10_digits,
    }
    for name, digits in baseline_predictions.items():
        baselines[name]["model_exact_agreement"] = float((preds == digits).all(axis=1).mean())
    print(json.dumps(baselines, indent=2))

    results = {
        "run_dir": str(run_dir), "checkpoint": ckpt_path, "n_val_examples": n,
        "sanity_check_mismatches": mismatches,
        "error_structure": {
            "bucket_counts": bucket_counts, "first_wrong_position_histogram": dict(first_wrong_hist),
            "longest_error_run_histogram": dict(Counter(error_run_lengths)),
            "frac_contiguous_given_wrong": frac_contiguous,
        },
        "per_position": per_position,
        "feature_bucket_report": feature_bucket_report,
        "baselines": baselines,
    }
    out_path = out_dir / "task_b_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
