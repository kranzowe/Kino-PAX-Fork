# KinoPaxSTAR Algorithm Variants

GPU kinodynamic planners in this repo, spanning pure exploration (KPAX) → pure
optimization (KinoPaxPlus) → the hybrid "STAR" line that fuses both. All share the
KPAX parallel frontier-expansion core (propagate → update graph → update frontier) and
the Syclop-style region probability-of-acceptance (PoA) for deciding which sampled nodes
survive into the next frontier. Model 1 (6D double integrator); cost = cumulative
workspace path length root→goal.

Two axes separate the STAR variants. **Admission gate**: what filters a newly propagated
non-best node before it is inserted (nothing / goal-progress / cost). **Seeding (pSeed)**:
the probability that a node is admitted purely because it landed in a never-before-occupied
R2 sub-region — the `|| (!activeSubVertices[x1SubVertex])` disjunct in the propagate
kernels. That free pass is unconditional (pSeed = 1) in KPAX and most STAR variants,
annealed 0→1 in `KinoPaxSTARsparsefill`, and removed entirely (pSeed = 0) in
`KinoPaxSTARnoseed` and `KinoPaxSTARcostprunenoseed`. Every variant still *marks* R2
occupancy (`atomicExch(&activeSubVertices[...], 1)`), so coverage diagnostics stay valid;
only admission changes.

## Baselines
### KPAX (Kino-PAX)
Base explorer. Propagates every frontier node in parallel; a new node joins the next
frontier when the Syclop region PoA passes (`curand_uniform < vertexScore`) or it lands in
a never-active R2 sub-region. Dormant tree nodes are reactivated by the same Syclop roll
plus an `fAccept` boost that decays as the tree fills. Excellent coverage, no cost
awareness → long paths.

### PruneKPAX
KPAX + (1) a spatial-hash collision grid for faster obstacle checks, (2) an extra
goal-oriented admission gate, and (3) goal-progress pruning that rejects nodes regressing
away from the goal. Faster and more goal-directed than KPAX; still not cost-optimal.

### KinoPaxPlus
The optimizer. Tracks per-region min cost (`minCostsR1`) and cumulative root→goal cost
(`h_minCost_`, via `atomicMinFloat`), always keeps each region's lowest-cost node, and
runs a dormancy/cost pruning subroutine (`d_pruned_` + `treeInactiveIterations`) that
prunes nodes which are neither region-best nor recently useful (the "ancestor/dormant
pruning" pass, on in this benchmark). Optimizes cost but explores weakly, so costs stay
high in cluttered maps.

## STAR hybrids (KPAX exploration + KinoPaxPlus cost)
### KinoPaxSTARNoPrune
Simplest fusion. Each iteration it (a) always keeps each region's lowest-cost node in the
frontier (KinoPaxPlus) and (b) also admits Syclop-accepted explorer nodes (KPAX) —
dual acceptance = `isBest OR acceptedByExploration`. Tracks `h_minCost_`; reactivates
dormant nodes with the plain Syclop roll. No pruning. Uses the spatial-hash collision grid.

### KinoPaxSTARNoPruneNoSpatialHash  *(new)*
Identical to KinoPaxSTARNoPrune but with the spatial-hash grid removed — collision checks
use the direct `propagateAndCheck` path (brute-force over obstacles) instead of
`propagateAndCheckSpatialHash`. Same nodes accepted; only the collision-check cost differs,
isolating the spatial-hash speedup from search behavior.

### KinoPaxSTAR
KinoPaxSTARNoPrune + the PruneKPAX pruning methodology: a goal-progress prune kernel gates
newly propagated non-best nodes on greedy-toward-goal progress before insertion, and
dormant-node reactivation blends goal-progress PoA with the Syclop score. Region-best
nodes are exempt from pruning.

### KinoPaxSTARcostprune
Cost-first pruning variant: KinoPaxSTAR with the goal-progress gate replaced by a cost gate.
Both propagate kernels accumulate per-region cost statistics (`minCostsR1` / `maxCostsR1` /
`sumCostsR1` / `cntCostsR1`), and `_costPrune_kernel` then keeps each non-best candidate with
probability `costKeepProb` (`include/helper/helper.cuh`) — region-best nodes are exempt and
always inserted. Reactivation multiplies that same keep-probability into the Syclop roll
(`costProb * (fminf(vertexScore, acceptCap) + fAccept)`). Four host fields tune it, set in the
constructor and deliberately *not* touched by `resetPlanner` so a benchmark can override them
per run: `h_acceptCap_` caps the Syclop exploration roll (`fminf(vertexScore, cap)`, taming the
1.0 score that `Graph` assigns to never-visited regions), `h_costPruneExp_` is the exponent `k`,
`h_costPruneFloor_` is a floor on the keep-probability so cheap regions keep exploring, and
`h_costPruneNorm_` picks the normalization — 0 = min-ratio `(m/cost)^k`, 1 = min-max
`((M-cost)/(M-m))^k` (default), 2 = mean `min(1,(mean/cost)^k)`; 1 and 2 keep `k` meaningful as
the region min approaches 0. The benchmarks sweep `(h_acceptCap_, h_costPruneExp_)` and label
each point `cap{100*cap}` plus `_exp{100*exp}` when the exponent is not 1.0 — e.g.
`KinoPaxSTARcostprune_cap40`, `KinoPaxSTARcostprune_cap0_exp75`. Cost dominates node retention.

### KinoPaxSTARnoseed
KinoPaxSTARNoPrune with the R2 sub-region seeding free pass deleted: acceptance is the bare
Syclop roll (`curand_uniform < vertexScore`), so a node landing in a fresh sub-region gets no
special treatment. Equivalent to pSeed = 0. Coverage grows far more slowly and trees stay
smaller per iteration — the test of how much of KPAX's exploration is actually seeding.

### KinoPaxSTARsparsefill
The annealed middle ground between KinoPaxSTARNoPrune (pSeed = 1) and KinoPaxSTARnoseed
(pSeed = 0): seeding is gated on `curand_uniform < pSeed` with
`pSeed = min(1, h_itr_ / h_rampIters_)` (`h_rampIters_` = 100 by default), so early iterations
search sparsely on cost and later ones fill in coverage.

### KinoPaxSTARcostprunenoseed  *(new)*
KinoPaxSTARcostprune's cost machinery verbatim — same region statistics, same
`_costPrune_kernel`, same cost-blended reactivation, same `h_acceptCap_` / `h_costPruneExp_` /
`h_costPruneFloor_` / `h_costPruneNorm_` tunables — with the R2 seeding disjunct gated on a
constant host field `h_pSeed_` that defaults to `0.0f`. Setting `h_pSeed_ = 1.0f` recovers stock
KinoPaxSTARcostprune, which makes the pair a clean controlled comparison. `kinopaxstar_cost_benchmark.cu`
runs both over the same `(cap, exp)` grid and labels the noseed points
`KinoPaxSTARcostprune_noseed_cap{0,40,100}[_exp{50,75}]`. Isolates how much of the cost gate's
benefit depends on seeding still supplying coverage underneath it.

## Quick comparison
| Variant | Explore | Seeding (pSeed) | Cost-aware | Pruning | Spatial hash |
|---|---|---|---|---|---|
| KPAX | Syclop | 1 | — | — | — |
| PruneKPAX | ✔ | 1 | goal-dir | goal-progress | ✔ |
| KinoPaxPlus | weak | n/a | ✔ | dormancy/cost | — |
| KinoPaxSTARNoPrune | ✔ | 1 | ✔ | — | ✔ |
| KinoPaxSTARNoPruneNoSpatialHash | ✔ | 1 | ✔ | — | — |
| KinoPaxSTAR | ✔ | 1 | ✔ | goal-progress | ✔ |
| KinoPaxSTARcostprune | capped (`h_acceptCap_`) | 1 | ✔ | cost | ✔ |
| KinoPaxSTARnoseed | ✔ | 0 | ✔ | — | ✔ |
| KinoPaxSTARsparsefill | ✔ | ramp 0→1 | ✔ | — | ✔ |
| KinoPaxSTARcostprunenoseed | capped (`h_acceptCap_`) | 0 (`h_pSeed_`) | ✔ | cost | ✔ |
