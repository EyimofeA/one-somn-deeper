"""Task A error-structure / arithmetic-difficulty / baseline / per-position analysis.

Analyzes an already-trained checkpoint (CPU inference only -- no training,
so no GPU needed here). Covers spec sections 1-4:
  1. error structure (how many digits wrong, where, contiguity)
  2. arithmetic-difficulty features + interpretable model of failure
  3. non-neural baselines
  4. per-position accuracy/entropy/calibration/confusion/carry-conditioning

Writes diagnostics/analysis_out/task_a_analysis.json (all data the report/
artifact is built from) and prints summary tables.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import DiagnosticDataset, load_jsonl
from data.tokens import NUM_SQUARE_DIGITS, NUM_SQUARE_X_DIGITS
from train import build_model

RUN_DIR = Path("runs/square_transformer_50k")
OUT_DIR = Path("analysis_out")
OUT_DIR.mkdir(exist_ok=True)
W = NUM_SQUARE_DIGITS  # 12
XW = NUM_SQUARE_X_DIGITS  # 6


# ---------------------------------------------------------------------------
# arithmetic feature extraction (schoolbook long multiplication of x by itself)
# ---------------------------------------------------------------------------
def significant_digits(x: int) -> list[int]:
    return [int(c) for c in str(x)]  # MSB-first, no leading zeros


def carry_features(x: int) -> dict:
    d = significant_digits(x)[::-1]  # LSB-first for column math
    n = len(d)
    # column_sum[k] = sum over i+j=k of d[i]*d[j], k in [0, 2n-2]
    column_sum = [0] * (2 * n - 1)
    contributions = [0] * (2 * n - 1)  # number of (i,j) pairs landing in this column
    for i in range(n):
        for j in range(n):
            column_sum[i + j] += d[i] * d[j]
            contributions[i + j] += 1

    carry_in = [0] * (2 * n)
    carry_out = [0] * (2 * n)
    carry = 0
    for k in range(2 * n - 1):
        carry_in[k] = carry
        total = column_sum[k] + carry
        carry = total // 10
        carry_out[k] = carry
    carry_in[2 * n - 1] = carry  # final overflow digit(s), if any

    total_carries = sum(1 for c in carry_out if c > 0)
    max_carry = max(carry_out) if carry_out else 0
    longest_chain = 0
    cur = 0
    for c in carry_out:
        if c > 0:
            cur += 1
            longest_chain = max(longest_chain, cur)
        else:
            cur = 0

    nearest_pow10 = min(abs(x - 10**k) for k in range(0, 7))
    return {
        "n_digits_x": n,
        "n_digits_x2": len(str(x * x)),
        "n_nonzero_digits_x": sum(1 for v in d if v != 0),
        "n_repeated_digits_x": n - len(set(d)),
        "n_partial_products": n * n,
        "total_carries": total_carries,
        "longest_carry_chain": longest_chain,
        "max_carry_value": max_carry,
        "max_multi_term_positions": max(contributions),
        "dist_to_pow10": nearest_pow10,
        "magnitude_bucket": len(str(x)) - 1,  # 0..5 -> x in [10^b, 10^(b+1))
        "column_sum": column_sum,
        "contributions": contributions,
        "carry_out_by_column": carry_out,
    }


def align_column_feature_to_output(col_feature: list[int], n_sig: int) -> list[int]:
    """Map a length-(2n-1) column-indexed feature (LSB-first, n=n_sig significant
    digits) onto the fixed W=12 zero-padded MSB-first output positions."""
    out = [0] * W
    # output position p (MSB-first, width W) corresponds to column (W-1-p) in a
    # LSB-first, W-wide frame; the real number only occupies the low 2*n_sig-1
    # columns of that frame (rest are leading zeros with no real column stats).
    for col in range(len(col_feature)):
        p = W - 1 - col
        if 0 <= p < W:
            out[p] = col_feature[col]
    return out


# ---------------------------------------------------------------------------
# model loading + batched inference
# ---------------------------------------------------------------------------
def load_trained_model():
    cfg = yaml.safe_load((RUN_DIR / "config_used.yaml").read_text())
    train_ds = DiagnosticDataset(cfg["data"]["train"])
    model = build_model(cfg, max_seq_len=train_ds.max_len, task="square")
    model.load_state_dict(torch.load(RUN_DIR / "peak.pt", map_location="cpu"))
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(model, rows: list[dict], batch_size: int = 256):
    all_preds, all_probs = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        ids = torch.tensor([r["input_ids"] for r in chunk], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.bool)
        logits = model(ids, mask)[:, -W:, :]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        all_preds.append(preds.numpy())
        all_probs.append(probs.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_probs, axis=0)


def main() -> None:
    model, cfg = load_trained_model()
    rows = load_jsonl(cfg["data"]["val"])
    print(f"loaded {len(rows)} val_iid rows, model={RUN_DIR}")

    preds, probs = run_inference(model, rows)  # (N, 12), (N, 12, 10)
    targets = np.array([r["labels"][-W:] for r in rows])  # (N, 12)
    xs = np.array([r["x"] for r in rows])

    correct = preds == targets  # (N, 12) bool
    n = len(rows)

    # ---------------- SECTION 1: error structure ----------------
    n_wrong = (~correct).sum(axis=1)  # per-example wrong-digit count
    bucket_names = ["0 wrong", "1 wrong", "2-3 wrong", "4+ wrong"]
    bucket_counts = [
        int((n_wrong == 0).sum()),
        int((n_wrong == 1).sum()),
        int(((n_wrong >= 2) & (n_wrong <= 3)).sum()),
        int((n_wrong >= 4).sum()),
    ]

    first_wrong_pos = []
    for i in range(n):
        wrong_idx = np.where(~correct[i])[0]
        first_wrong_pos.append(int(wrong_idx[0]) if len(wrong_idx) else -1)
    first_wrong_pos = np.array(first_wrong_pos)
    first_wrong_hist = Counter(p for p in first_wrong_pos if p >= 0)

    # contagion: P(position p+1 wrong | position p wrong) vs P(position p+1 wrong) unconditional
    contagion = {}
    for p in range(W - 1):
        given_wrong = correct[:, p] == False
        if given_wrong.sum() > 0:
            p_next_wrong_given = float((~correct[given_wrong, p + 1]).mean())
        else:
            p_next_wrong_given = None
        p_next_wrong_uncond = float((~correct[:, p + 1]).mean())
        contagion[p] = {"p_next_wrong_given_this_wrong": p_next_wrong_given, "p_next_wrong_unconditional": p_next_wrong_uncond}

    # contiguity: for wrong examples, is the wrong-position set one contiguous run?
    def is_contiguous(wrong_positions: np.ndarray) -> bool:
        if len(wrong_positions) <= 1:
            return True
        return bool(np.all(np.diff(wrong_positions) == 1))

    contiguous_count = 0
    wrong_example_count = 0
    for i in range(n):
        wp = np.where(~correct[i])[0]
        if len(wp) > 0:
            wrong_example_count += 1
            if is_contiguous(wp):
                contiguous_count += 1
    frac_contiguous = contiguous_count / wrong_example_count if wrong_example_count else None

    # 20 representative failures grouped by error pattern (bucket by n_wrong)
    representative_failures = []
    for target_n_wrong in (1, 2, 3, 5, 8, 12):
        idxs = np.where(n_wrong == target_n_wrong)[0]
        for idx in idxs[:4]:
            representative_failures.append({
                "x": int(xs[idx]),
                "target": int(xs[idx]) ** 2,
                "true_digits": targets[idx].tolist(),
                "pred_digits": preds[idx].tolist(),
                "wrong_positions": np.where(~correct[idx])[0].tolist(),
                "n_wrong": int(target_n_wrong),
            })
        if len(representative_failures) >= 20:
            break

    section1 = {
        "bucket_names": bucket_names,
        "bucket_counts": bucket_counts,
        "bucket_fractions": [c / n for c in bucket_counts],
        "first_wrong_position_histogram": {int(k): int(v) for k, v in sorted(first_wrong_hist.items())},
        "contagion_by_position": contagion,
        "frac_contiguous_given_wrong": frac_contiguous,
        "n_wrong_examples": wrong_example_count,
        "representative_failures": representative_failures[:20],
    }
    print("\n=== SECTION 1: error structure ===")
    print(f"exact match: {bucket_counts[0]}/{n} = {bucket_counts[0]/n:.4f}")
    for name, c in zip(bucket_names, bucket_counts):
        print(f"  {name}: {c} ({c/n:.4f})")
    print(f"fraction of wrong examples whose wrong positions are one contiguous run: {frac_contiguous:.4f}" if frac_contiguous else "n/a")

    # ---------------- SECTION 2: arithmetic difficulty ----------------
    print("\n=== SECTION 2: arithmetic difficulty ===")
    feats = [carry_features(int(x)) for x in xs]
    feature_names = [
        "n_digits_x", "n_digits_x2", "n_nonzero_digits_x", "n_repeated_digits_x",
        "n_partial_products", "total_carries", "longest_carry_chain", "max_carry_value",
        "max_multi_term_positions", "dist_to_pow10", "magnitude_bucket",
    ]
    X_feat = np.array([[f[name] for name in feature_names] for f in feats], dtype=float)
    y_exact = (n_wrong == 0).astype(int)
    y_token_acc = correct.mean(axis=1)

    bucket_report = {}
    for name in feature_names:
        vals = X_feat[:, feature_names.index(name)]
        # bucket into quantiles (or unique small-cardinality values)
        uniq = np.unique(vals)
        if len(uniq) <= 8:
            buckets = uniq
            assign = vals
        else:
            edges = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
            buckets = edges
            assign = np.digitize(vals, edges[1:-1])
        rep = {}
        for b in np.unique(assign):
            mask = assign == b
            rep[str(b)] = {"n": int(mask.sum()), "exact_match": float(y_exact[mask].mean()), "token_accuracy": float(y_token_acc[mask].mean())}
        bucket_report[name] = rep
    print("feature -> exact-match by bucket (n):")
    for name, rep in bucket_report.items():
        print(f"  {name}: " + ", ".join(f"{k}:{v['exact_match']:.3f}(n={v['n']})" for k, v in rep.items()))

    # interpretable model: logistic regression + shallow decision tree predicting exact-match
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    Xtr, Xte, ytr, yte = train_test_split(X_feat, y_exact, test_size=0.2, random_state=0, stratify=y_exact)
    scaler = StandardScaler().fit(Xtr)
    logreg = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    logreg_acc = logreg.score(scaler.transform(Xte), yte)
    logreg_coefs = dict(zip(feature_names, logreg.coef_[0].tolist()))

    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr)
    tree_acc = tree.score(Xte, yte)
    tree_importances = dict(zip(feature_names, tree.feature_importances_.tolist()))
    tree_text = export_text(tree, feature_names=feature_names)

    print(f"\nlogreg test accuracy predicting exact-match: {logreg_acc:.4f} (majority baseline: {max(yte.mean(), 1-yte.mean()):.4f})")
    print("logreg standardized coefficients (feature importance direction):")
    for k, v in sorted(logreg_coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v:+.4f}")
    print(f"\ndecision tree test accuracy: {tree_acc:.4f}")
    print("decision tree feature importances:")
    for k, v in sorted(tree_importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v:.4f}")

    section2 = {
        "feature_names": feature_names,
        "bucket_report": bucket_report,
        "logreg_test_accuracy": logreg_acc,
        "logreg_majority_baseline": float(max(yte.mean(), 1 - yte.mean())),
        "logreg_coefficients": logreg_coefs,
        "tree_test_accuracy": tree_acc,
        "tree_feature_importances": tree_importances,
        "tree_text": tree_text,
    }

    # ---------------- SECTION 3: baselines ----------------
    print("\n=== SECTION 3: non-neural baselines ===")
    train_rows = load_jsonl(cfg["data"]["train"])
    train_targets = np.array([r["labels"][-W:] for r in train_rows])
    train_xs = np.array([r["x"] for r in train_rows])

    # (a) most common digit independently per position
    mode_digit_per_pos = np.array([Counter(train_targets[:, p]).most_common(1)[0][0] for p in range(W)])
    baseline_mode = (mode_digit_per_pos[None, :] == targets)
    exact_mode = baseline_mode.all(axis=1).mean()
    token_mode = baseline_mode.mean()

    # (b) copy input digits into output (simple positional heuristic: does x's own digits ever equal x^2's digits at same position?)
    x_digit_seqs = np.array([r["input_ids"][2 : 2 + XW] for r in rows]) - 9  # raw x digits, MSB-first, width 6
    # naive heuristic: pad x's 6 digits to the right to fill 12 slots (copy input straight to output, one popular "shortcut" a undertrained net might take)
    copy_pred = np.zeros((n, W), dtype=int)
    copy_pred[:, -XW:] = x_digit_seqs
    baseline_copy = copy_pred == targets
    exact_copy = baseline_copy.all(axis=1).mean()
    token_copy = baseline_copy.mean()

    # (c) nearest training target by input value |x - x_train| minimized, use that x_train^2's digits
    train_xs_sorted_idx = np.argsort(train_xs)
    train_xs_sorted = train_xs[train_xs_sorted_idx]
    nearest_val_idx = np.searchsorted(train_xs_sorted, xs)
    nearest_val_idx = np.clip(nearest_val_idx, 0, len(train_xs_sorted) - 1)
    # check neighbor on both sides, take closer one
    left_idx = np.clip(nearest_val_idx - 1, 0, len(train_xs_sorted) - 1)
    use_left = np.abs(train_xs_sorted[left_idx] - xs) < np.abs(train_xs_sorted[nearest_val_idx] - xs)
    chosen = np.where(use_left, left_idx, nearest_val_idx)
    nearest_train_orig_idx = train_xs_sorted_idx[chosen]
    baseline_nearest_val = train_targets[nearest_train_orig_idx] == targets
    exact_nearest_val = baseline_nearest_val.all(axis=1).mean()
    token_nearest_val = baseline_nearest_val.mean()

    # (d) nearest training input by digit Hamming distance (over the 6 raw x digits)
    train_x_digit_seqs = np.array([r["input_ids"][2 : 2 + XW] for r in train_rows]) - 9
    # sample a subset of val rows for this O(n_val * n_train) baseline (expensive at full scale)
    sample_n = min(500, n)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    hamming_correct = np.zeros(sample_n, dtype=bool)
    for si, i in enumerate(sample_idx):
        dists = (train_x_digit_seqs != x_digit_seqs[i]).sum(axis=1)
        best = np.argmin(dists)
        hamming_correct[si] = np.array_equal(train_targets[best], targets[i])
    exact_hamming = hamming_correct.mean()

    # (e) lookup table for exact seen inputs (val_iid is held-out x by construction -> expect 0)
    train_x_set = set(train_xs.tolist())
    exact_lookup = float(np.mean([1 if int(x) in train_x_set else 0 for x in xs]))

    # (f) leading/trailing digit pattern only: predict correct only the leading digit / trailing digit, rest from mode
    leading_only_correct = correct[:, 0].mean()
    trailing_only_correct = correct[:, -1].mean()

    section3 = {
        "most_common_digit_per_position": {"exact_match": float(exact_mode), "token_accuracy": float(token_mode)},
        "copy_input_digits_heuristic": {"exact_match": float(exact_copy), "token_accuracy": float(token_copy)},
        "nearest_training_target_by_value": {"exact_match": float(exact_nearest_val), "token_accuracy": float(token_nearest_val)},
        "nearest_training_input_by_hamming": {"exact_match": float(exact_hamming), "n_sampled": sample_n},
        "lookup_table_exact_seen_inputs": {"fraction_of_val_seen_in_train": exact_lookup},
        "leading_digit_only_accuracy": float(leading_only_correct),
        "trailing_digit_only_accuracy": float(trailing_only_correct),
        "trained_model": {"exact_match": float(bucket_counts[0] / n), "token_accuracy": float(correct.mean())},
    }
    print(json.dumps(section3, indent=2))

    # ---------------- SECTION 4: per-position analysis ----------------
    print("\n=== SECTION 4: per-position analysis ===")
    per_position = []
    for p in range(W):
        acc = float(correct[:, p].mean())
        true_dist = np.bincount(targets[:, p], minlength=10) / n
        true_entropy = float(-(true_dist[true_dist > 0] * np.log2(true_dist[true_dist > 0])).sum())
        pred_dist = probs[:, p, :].mean(axis=0)
        pred_entropy = float(-(pred_dist[pred_dist > 0] * np.log2(pred_dist[pred_dist > 0])).sum())
        # calibration: mean confidence of argmax vs accuracy
        conf = probs[:, p, :].max(axis=1)
        calibration_gap = float(conf.mean() - acc)
        # confusion pairs
        wrong_mask = ~correct[:, p]
        confusions = Counter(zip(targets[wrong_mask, p].tolist(), preds[wrong_mask, p].tolist()))
        top_confusions = confusions.most_common(3)
        per_position.append({
            "position": p, "accuracy": acc, "true_entropy_bits": true_entropy,
            "pred_entropy_bits": pred_entropy, "mean_confidence": float(conf.mean()),
            "calibration_gap": calibration_gap, "top_confusions": top_confusions,
        })

    # carry-conditioned accuracy per position: build per-example, per-output-position "carry enters here" flag
    carry_enters = np.zeros((n, W), dtype=bool)
    n_terms_at_pos = np.zeros((n, W), dtype=int)
    for i, f in enumerate(feats):
        n_sig = f["n_digits_x"]
        col_carry_in = [0] + f["carry_out_by_column"][:-1]  # carry INTO column k = carry OUT of column k-1
        aligned_carry = align_column_feature_to_output([1 if c > 0 else 0 for c in col_carry_in], n_sig)
        aligned_terms = align_column_feature_to_output(f["contributions"], n_sig)
        carry_enters[i] = aligned_carry
        n_terms_at_pos[i] = aligned_terms

    for p in range(W):
        with_carry = carry_enters[:, p]
        acc_with_carry = float(correct[with_carry, p].mean()) if with_carry.sum() > 0 else None
        acc_without_carry = float(correct[~with_carry, p].mean()) if (~with_carry).sum() > 0 else None
        per_position[p]["frac_examples_with_carry_in"] = float(with_carry.mean())
        per_position[p]["accuracy_given_carry_in"] = acc_with_carry
        per_position[p]["accuracy_given_no_carry_in"] = acc_without_carry

        terms = n_terms_at_pos[:, p]
        by_terms = {}
        for t in np.unique(terms):
            mask = terms == t
            if mask.sum() > 5:
                by_terms[int(t)] = {"n": int(mask.sum()), "accuracy": float(correct[mask, p].mean())}
        per_position[p]["accuracy_by_n_multiplicative_terms"] = by_terms

    print(f"{'pos':>3} {'acc':>7} {'true_H':>7} {'pred_H':>7} {'conf':>6} {'calib_gap':>9} {'acc|carry':>9} {'acc|no_carry':>12}")
    for r in per_position:
        awc = r["accuracy_given_carry_in"]
        anc = r["accuracy_given_no_carry_in"]
        print(f"{r['position']:>3} {r['accuracy']:>7.4f} {r['true_entropy_bits']:>7.3f} {r['pred_entropy_bits']:>7.3f} "
              f"{r['mean_confidence']:>6.3f} {r['calibration_gap']:>+9.4f} "
              f"{'n/a' if awc is None else f'{awc:.4f}':>9} {'n/a' if anc is None else f'{anc:.4f}':>12}")

    section4 = {"per_position": per_position}

    # ---------------- write everything out ----------------
    results = {
        "run_dir": str(RUN_DIR),
        "n_val_examples": n,
        "section1_error_structure": section1,
        "section2_arithmetic_difficulty": section2,
        "section3_baselines": section3,
        "section4_per_position": section4,
    }
    out_path = OUT_DIR / "task_a_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
