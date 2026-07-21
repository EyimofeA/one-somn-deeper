# Ideas backlog

Three columns so we do not mix taste, evidence, and literature. Append-only; mark status when we try something.

## Yours (principal)

| Idea | Why | Status |
|------|-----|--------|
| Learned / adaptive T (or early stop) so compute follows T | Fixed K wastes or under-spends | queued — after K-sweep |
| Must generalize mod across many N, not one fixed N | e5 collapse | active constraint |
| Efficiency under 60s clock | Always | active |

## Ours (from measurements)

| Idea | Evidence | Status |
|------|----------|--------|
| Small width + tied loops beat wide shallow on Easy e1 | d32×K4 = 5.5% vs d64 K1 = 1.3% | adopted as reference |
| Always gate on e1 **and** e5 | 5.5% → 0.79% same weights | adopted |
| K-sweep at d=32 before adaptive halt | Only know K=1 vs 4 well at d=32 | **in progress** |
| N-conditioning inside the same looped block | e5 needs many rings ℤ/Nℤ | next after K-sweep |

## Researched (prior art)

| Idea | Pointer | Status |
|------|---------|--------|
| Weight-tied block + ACT / halt | Universal Transformer; Graves ACT | candidate after fixed-K sweep |
| Extra loops at eval for harder inputs | Looped / recurrent-depth Transformers | candidate |
| Iterative latent updates for algorithms | Algorithmic reasoning / vertical CoT notes | background |

## Not doing (for now)

- Untied deep stacks like an LLM (burns params, wrong bias for “same step again”)
- Hard/Medium until e5 mean moves clearly
- Treating e1-only wins as done
