# Model-review experiment ledger

No experiment in this ledger is launched solely from a model review. Entries are proposals to be ranked after the human reads one selected review.

| Proposal | Central change | Control | Success criterion | Kill condition | Status |
|---|---|---|---|---|---|
| Clean latent recurrent workspace | Encode `(N,x)` to latent `h0`; apply one tied learned transition exactly T times; decode only at the end | Per-position register control, same small-N final-label data | Unseen-N T=1 17.29% vs 9.35% control; T=8 14.49% vs 14.02% | Not yet promoted: larger-scale unseen-N or composition control fails | Completed: narrow positive signal; artifact `diagnostics/artifacts/clean_latent_workspace_seed0/eval_report.json` |
| Transition-supervised diagnostic bridge | Use controlled trace supervision to test whether latent transition can represent Square→Reduce | Final-label-only latent control | Large held-out transition and composed-depth gain | Trace fit without held-out transition gain | Diagnostic only; legality review required |
| RNS-channel representation | Learned mixed-radix/residue workspace with no hard-coded solver | Digit-register and latent controls | Cross-modulus T=1 gain under same wall-clock budget | No OOD-N gain or any legality violation | Unranked review proposal |
| Tied token-register iterator | Existing Fable T-cap + AdamW baseline | Current local Easy/Medium revalidation | Beat pre-registered promotion gate with T=1 certificate | Fails held-out gate | Completed/refuted locally |

References are opinions in `claude code fable/FULL_TRANSCRIPT.md`; project evidence is cited in the status and research logs. No prompt-tail curriculum tuning is proposed here.
