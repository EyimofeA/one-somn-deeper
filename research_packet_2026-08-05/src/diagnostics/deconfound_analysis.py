"""Deconfounded follow-up to analysis_task_a.py.

The first pass (analysis_task_a.py) showed accuracy crashes at output
positions 4-7 and is lower on examples with an incoming carry at those
positions -- but position, carry, and term-count all co-vary (central columns
of a schoolbook square have the most contributing digit pairs AND the largest
carries), so that alone cannot separate "carry propagation is the bottleneck"
from "dense partial-product aggregation is the bottleneck". This script:

  1. computes true per-column diagonal sum / term counts / carry in&out /
     output digit for every val_iid example, aligned to the 12 fixed output
     positions (LSB-first arithmetic, MSB-first output slots -- see
     analysis_task_a.align_column_feature_to_output)
  2. builds joint accuracy tables (position x carry-in VALUE, position x
     n-terms, position x diagonal-sum bucket) and controlled comparisons that
     hold one variable fixed while varying another
  3. fits a per-digit error model (class-balanced logistic regression) over
     every (example, position) pair, reporting PR-AUC/ROC-AUC/balanced
     accuracy/coefficients -- not plain accuracy, which is useless when
     positive (correct) or negative (wrong) is rare depending on position
  4. tests the "two-sided approximation" hypothesis: prefix-k / suffix-k
     exact-match, k=1..6, and the width of the correct prefix/suffix per
     example
  5. layerwise linear probes (diagonal sum, carry-in, carry-out, leading
     digit, x^2 mod 10) at each of the 4 encoder layers' hidden states

Writes analysis_out/task_a_deconfound.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis_task_a import (
    OUT_DIR,
    RUN_DIR,
    W,
    align_column_feature_to_output,
    carry_features,
    load_trained_model,
    run_inference,
)
from data.dataset import load_jsonl

N_SAMPLE = 10_000  # val_iid.jsonl has exactly this many rows


def build_per_position_arrays(xs: np.ndarray) -> dict[str, np.ndarray]:
    """For every example, align the true schoolbook quantities to the 12
    fixed (MSB-first) output positions. Shapes are all (n_examples, 12)."""
    n = len(xs)
    diag_sum = np.zeros((n, W), dtype=float)
    n_terms = np.zeros((n, W), dtype=int)
    n_nonzero_terms = np.zeros((n, W), dtype=int)
    carry_in_val = np.zeros((n, W), dtype=int)
    carry_out_val = np.zeros((n, W), dtype=int)
    digit_result = np.zeros((n, W), dtype=int)

    for i, x in enumerate(xs):
        f = carry_features(int(x))
        n_sig = f["n_digits_x"]
        diag_sum[i] = align_column_feature_to_output(f["column_sum"], n_sig)
        n_terms[i] = align_column_feature_to_output(f["contributions"], n_sig)
        n_nonzero_terms[i] = align_column_feature_to_output(f["nonzero_contributions"], n_sig)
        carry_in_val[i] = align_column_feature_to_output(f["carry_in_by_column"], n_sig)
        carry_out_val[i] = align_column_feature_to_output(f["carry_out_by_column"], n_sig)
        digit_result[i] = align_column_feature_to_output(f["digit_result"], n_sig)

    return {
        "diag_sum": diag_sum, "n_terms": n_terms, "n_nonzero_terms": n_nonzero_terms,
        "carry_in": carry_in_val, "carry_out": carry_out_val, "digit_result": digit_result,
    }


def bucket_edges(values: np.ndarray, n_buckets: int = 4) -> np.ndarray:
    qs = np.linspace(0, 1, n_buckets + 1)
    return np.unique(np.quantile(values, qs))


def joint_table(correct: np.ndarray, key_a: np.ndarray, key_b: np.ndarray, min_n: int = 20) -> dict:
    """accuracy of `correct` (bool array), grouped by (key_a, key_b) pairs."""
    out: dict[str, dict[str, dict]] = {}
    for a in np.unique(key_a):
        row = {}
        for b in np.unique(key_b[key_a == a]):
            mask = (key_a == a) & (key_b == b)
            if mask.sum() >= min_n:
                row[str(b)] = {"n": int(mask.sum()), "accuracy": float(correct[mask].mean())}
        if row:
            out[str(a)] = row
    return out


def main() -> None:
    model, cfg = load_trained_model()
    rows = load_jsonl(cfg["data"]["val"])
    assert len(rows) == N_SAMPLE
    xs = np.array([r["x"] for r in rows])
    preds, probs = run_inference(model, rows)
    targets = np.array([r["labels"][-W:] for r in rows])
    correct = preds == targets  # (N, 12)
    n = len(rows)

    print("Computing true per-column schoolbook quantities for all", n, "examples...")
    arrs = build_per_position_arrays(xs)

    # sanity check: the schoolbook simulation's own digit_result must exactly
    # equal the true label at every position (if not, the feature extraction
    # itself is wrong and nothing downstream can be trusted)
    mismatch = (arrs["digit_result"] != targets).sum()
    print(f"sanity check: digit_result vs true target mismatches = {mismatch} / {targets.size} "
          f"(must be 0 -- confirms the carry simulation reproduces x*x exactly)")
    assert mismatch == 0, "carry_features' simulated digits do not match true x*x digits -- bug in feature extraction"

    # ---------------- confound check: how much do carry/term-count/diag-sum
    # actually co-vary with position? ----------------
    print("\n=== Confound check: do carry-in, n_terms, diag_sum all rise toward the middle positions? ===")
    confound_summary = {}
    for p in range(W):
        confound_summary[p] = {
            "mean_carry_in": float(arrs["carry_in"][:, p].mean()),
            "mean_n_terms": float(arrs["n_terms"][:, p].mean()),
            "mean_diag_sum": float(arrs["diag_sum"][:, p].mean()),
            "accuracy": float(correct[:, p].mean()),
        }
        r = confound_summary[p]
        print(f"  pos {p}: mean carry_in={r['mean_carry_in']:.2f} mean_n_terms={r['mean_n_terms']:.2f} "
              f"mean_diag_sum={r['mean_diag_sum']:.1f} accuracy={r['accuracy']:.4f}")

    # ---------------- joint accuracy tables ----------------
    print("\n=== Joint accuracy: position x carry-in VALUE ===")
    pos_grid = np.tile(np.arange(W), (n, 1))  # (n, 12), position index broadcast
    jt_pos_carry = joint_table(correct.ravel(), pos_grid.ravel(), arrs["carry_in"].ravel(), min_n=20)
    for p in sorted(jt_pos_carry, key=lambda k: int(k)):
        print(f"  pos {p}: " + ", ".join(f"carry={c}:{v['accuracy']:.3f}(n={v['n']})" for c, v in sorted(jt_pos_carry[p].items(), key=lambda kv: int(kv[0]))))

    print("\n=== Joint accuracy: position x number of contributing terms ===")
    jt_pos_terms = joint_table(correct.ravel(), pos_grid.ravel(), arrs["n_terms"].ravel(), min_n=20)
    for p in sorted(jt_pos_terms, key=lambda k: int(k)):
        print(f"  pos {p}: " + ", ".join(f"terms={c}:{v['accuracy']:.3f}(n={v['n']})" for c, v in sorted(jt_pos_terms[p].items(), key=lambda kv: int(kv[0]))))

    # diagonal sum bucketed (continuous -> quantile buckets)
    diag_flat = arrs["diag_sum"].ravel()
    diag_edges = bucket_edges(diag_flat, 4)
    diag_bucket = np.digitize(diag_flat, diag_edges[1:-1])
    print("\n=== Joint accuracy: position x diagonal-sum bucket (quantile-binned) ===")
    jt_pos_diag = joint_table(correct.ravel(), pos_grid.ravel(), diag_bucket, min_n=20)
    for p in sorted(jt_pos_diag, key=lambda k: int(k)):
        print(f"  pos {p}: " + ", ".join(f"diagQ{c}:{v['accuracy']:.3f}(n={v['n']})" for c, v in sorted(jt_pos_diag[p].items(), key=lambda kv: int(kv[0]))))

    # ---------------- controlled comparisons ----------------
    print("\n=== Controlled comparison A: fix position + diagonal-sum bucket, vary carry-in ===")
    controlled_a = {}
    for target_pos in (4, 5, 6, 7):
        for target_diagq in (1, 2):  # mid-range diagonal-sum buckets, most populated
            mask_base = (pos_grid.ravel() == target_pos) & (diag_bucket == target_diagq)
            if mask_base.sum() < 100:
                continue
            row = {}
            for c in np.unique(arrs["carry_in"].ravel()[mask_base]):
                mask = mask_base & (arrs["carry_in"].ravel() == c)
                if mask.sum() >= 15:
                    row[int(c)] = {"n": int(mask.sum()), "accuracy": float(correct.ravel()[mask].mean())}
            if len(row) >= 2:
                key = f"pos={target_pos},diagQ={target_diagq}"
                controlled_a[key] = row
                print(f"  {key}: " + ", ".join(f"carry_in={c}:{v['accuracy']:.3f}(n={v['n']})" for c, v in sorted(row.items())))

    print("\n=== Controlled comparison B: fix position + carry-in bucket, vary n_terms ===")
    carry_flat = arrs["carry_in"].ravel()
    carry_bucket = np.clip(carry_flat, 0, 3)  # 0,1,2,3+ -- carries beyond 3 are rare here
    controlled_b = {}
    for target_pos in (4, 5, 6, 7):
        for target_carryb in (0, 1):
            mask_base = (pos_grid.ravel() == target_pos) & (carry_bucket == target_carryb)
            if mask_base.sum() < 100:
                continue
            row = {}
            for t in np.unique(arrs["n_terms"].ravel()[mask_base]):
                mask = mask_base & (arrs["n_terms"].ravel() == t)
                if mask.sum() >= 15:
                    row[int(t)] = {"n": int(mask.sum()), "accuracy": float(correct.ravel()[mask].mean())}
            if len(row) >= 2:
                key = f"pos={target_pos},carry_in={target_carryb}"
                controlled_b[key] = row
                print(f"  {key}: " + ", ".join(f"n_terms={t}:{v['accuracy']:.3f}(n={v['n']})" for t, v in sorted(row.items())))

    print("\n=== Controlled comparison C: fix position + term-count bucket, vary carry MAGNITUDE ===")
    controlled_c = {}
    for target_pos in (4, 5, 6, 7):
        terms_at_pos = arrs["n_terms"][:, target_pos]
        modal_terms = int(np.bincount(terms_at_pos).argmax())
        mask_base = (pos_grid.ravel() == target_pos) & (arrs["n_terms"].ravel() == modal_terms)
        if mask_base.sum() < 100:
            continue
        row = {}
        for c in np.unique(carry_flat[mask_base]):
            mask = mask_base & (carry_flat == c)
            if mask.sum() >= 15:
                row[int(c)] = {"n": int(mask.sum()), "accuracy": float(correct.ravel()[mask].mean())}
        if len(row) >= 2:
            key = f"pos={target_pos},n_terms={modal_terms}(modal)"
            controlled_c[key] = row
            print(f"  {key}: " + ", ".join(f"carry_in={c}:{v['accuracy']:.3f}(n={v['n']})" for c, v in sorted(row.items())))

    # ---------------- per-digit error model ----------------
    print("\n=== Per-digit error model (class-balanced logistic regression) ===")
    feat_names = ["position", "diag_sum", "n_terms", "n_nonzero_terms", "carry_in", "carry_out"]
    X = np.stack([
        pos_grid.ravel().astype(float), arrs["diag_sum"].ravel(), arrs["n_terms"].ravel().astype(float),
        arrs["n_nonzero_terms"].ravel().astype(float), arrs["carry_in"].ravel().astype(float),
        arrs["carry_out"].ravel().astype(float),
    ], axis=1)
    y = correct.ravel().astype(int)  # 1 = correct, 0 = wrong

    # subsample for tractable fitting (12 * 10000 = 120,000 rows; fine, but keep a held-out test split)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    proba_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
    pred_te = clf.predict(scaler.transform(Xte))

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred_te)),
        "roc_auc": float(roc_auc_score(yte, proba_te)),
        "pr_auc": float(average_precision_score(yte, proba_te)),
        "positive_rate_test": float(yte.mean()),
    }
    coefs = dict(zip(feat_names, clf.coef_[0].tolist()))
    print(json.dumps(metrics, indent=2))
    print("standardized coefficients (predicting P(correct)):")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v:+.4f}")

    per_digit_model = {"feature_names": feat_names, "test_metrics": metrics, "coefficients": coefs}

    # ---------------- Hypothesis C: two-sided approximation ----------------
    print("\n=== Hypothesis C check: prefix-k / suffix-k exact match, k=1..6 ===")
    prefix_exact = {}
    suffix_exact = {}
    for k in range(1, 7):
        prefix_exact[k] = float(correct[:, :k].all(axis=1).mean())
        suffix_exact[k] = float(correct[:, -k:].all(axis=1).mean())
        print(f"  k={k}: prefix exact-match={prefix_exact[k]:.4f}  suffix exact-match={suffix_exact[k]:.4f}")

    # correct-prefix width = largest k (0..12) such that positions [0,k) are ALL correct
    prefix_width = np.zeros(n, dtype=int)
    suffix_width = np.zeros(n, dtype=int)
    for i in range(n):
        k = 0
        while k < W and correct[i, k]:
            k += 1
        prefix_width[i] = k
        k = 0
        while k < W and correct[i, W - 1 - k]:
            k += 1
        suffix_width[i] = k
    reconciliation_gap = np.clip(W - prefix_width - suffix_width, 0, W)
    gap_hist = {int(g): int(c) for g, c in zip(*np.unique(reconciliation_gap, return_counts=True))}
    print(f"\nprefix_width mean={prefix_width.mean():.2f}  suffix_width mean={suffix_width.mean():.2f}")
    print(f"reconciliation gap (12 - prefix_width - suffix_width, clipped>=0) histogram: {gap_hist}")
    print("If hypothesis C (independent high/low regions failing only to reconcile in the middle) were the whole "
          "story, prefix_width+suffix_width would cluster near 12 with a small, narrow gap. If failures are instead "
          "concentrated at a small number of FIXED positions regardless of how far the correct region extends, the "
          "gap should cluster tightly around a constant (~4, matching positions 4-7) rather than varying per example.")

    hypothesis_c = {
        "prefix_exact_by_k": prefix_exact, "suffix_exact_by_k": suffix_exact,
        "prefix_width_mean": float(prefix_width.mean()), "suffix_width_mean": float(suffix_width.mean()),
        "reconciliation_gap_histogram": gap_hist,
    }

    # ---------------- layerwise linear probes ----------------
    print("\n=== Layerwise linear probes ===")
    probe_n = min(4000, n)
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(n, size=probe_n, replace=False)
    probe_rows = [rows[i] for i in probe_idx]
    hidden_by_layer = collect_layer_hidden_states(model, probe_rows)  # list of (probe_n, seq_len, d_model)

    probe_results = {}
    for layer_idx, hidden in enumerate(hidden_by_layer):
        layer_probe = {}
        out_hidden = hidden[:, -W:, :]  # (probe_n, 12, d_model)
        for name, target_arr, kind in [
            ("diag_sum", arrs["diag_sum"][probe_idx], "regression"),
            ("carry_in", arrs["carry_in"][probe_idx], "regression"),
            ("carry_out", arrs["carry_out"][probe_idx], "regression"),
        ]:
            # probe each output position's hidden state against that position's quantity, pooled across positions 4-7
            r2s = []
            for p in (4, 5, 6, 7):
                Xp = out_hidden[:, p, :]
                yp = target_arr[:, p]
                if yp.std() < 1e-6:
                    continue
                Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.3, random_state=0)
                ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
                r2 = ridge.score(Xte, yte)
                r2s.append(r2)
            layer_probe[name] = float(np.mean(r2s)) if r2s else None

        # global quantities: probe from position 0's hidden state (leading digit) and position 11's (mod 10)
        leading_digit = np.array([int(str(int(x))[0]) for x in xs[probe_idx]])
        mod10 = (xs[probe_idx].astype(np.int64) ** 2) % 10
        Xp0 = out_hidden[:, 0, :]
        Xtr, Xte, ytr, yte = train_test_split(Xp0, leading_digit, test_size=0.3, random_state=0)
        clf_lead = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        layer_probe["leading_digit_accuracy"] = float(clf_lead.score(Xte, yte))

        Xp11 = out_hidden[:, 11, :]
        Xtr, Xte, ytr, yte = train_test_split(Xp11, mod10, test_size=0.3, random_state=0)
        clf_mod = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        layer_probe["x2_mod_10_accuracy"] = float(clf_mod.score(Xte, yte))

        probe_results[f"layer_{layer_idx}"] = layer_probe
        print(f"  layer {layer_idx}: " + ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in layer_probe.items()))

    # ---------------- write everything ----------------
    results = {
        "run_dir": str(RUN_DIR),
        "n_examples": n,
        "sanity_check_digit_result_mismatches": int(mismatch),
        "confound_summary_by_position": confound_summary,
        "joint_position_x_carry_in": jt_pos_carry,
        "joint_position_x_n_terms": jt_pos_terms,
        "joint_position_x_diag_sum_bucket": jt_pos_diag,
        "controlled_A_fix_pos_and_diagsum_vary_carry": controlled_a,
        "controlled_B_fix_pos_and_carry_vary_n_terms": controlled_b,
        "controlled_C_fix_pos_and_n_terms_vary_carry_magnitude": controlled_c,
        "per_digit_error_model": per_digit_model,
        "hypothesis_c_two_sided_approximation": hypothesis_c,
        "layerwise_probes": probe_results,
    }
    out_path = OUT_DIR / "task_a_deconfound.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


@torch.no_grad()
def collect_layer_hidden_states(model, rows: list[dict], batch_size: int = 256) -> list[np.ndarray]:
    """Runs the StandardTransformer manually, layer by layer, returning the
    residual-stream hidden state after each encoder layer (before the final
    norm/head). len(result) == n_layers."""
    n_layers = len(model.encoder.layers)
    all_layers = [[] for _ in range(n_layers)]
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        ids = torch.tensor([r["input_ids"] for r in chunk], dtype=torch.long)
        seq_len = ids.shape[1]
        positions = torch.arange(seq_len)
        h = model.token_embed(ids) + model.pos_embed(positions)[None, :, :]
        for li, layer in enumerate(model.encoder.layers):
            h = layer(h)
            all_layers[li].append(h.numpy())
    return [np.concatenate(layer_chunks, axis=0) for layer_chunks in all_layers]


if __name__ == "__main__":
    main()
