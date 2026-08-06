# KinoPaxSTAR Algorithm Variants

GPU kinodynamic planners in this repo, spanning pure exploration (KPAX) → pure
optimization (KinoPaxPlus) → the hybrid "STAR" line that fuses both. All share the
KPAX parallel frontier-expansion core (propagate → update graph → update frontier) and
the Syclop-style region probability-of-acceptance (PoA) for deciding which sampled nodes
survive into the next frontier. Model 1 (6D double integrator); cost = cumulative
workspace path length root→goal.

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
Cost-first pruning variant. Same dual-track admission, but Syclop exploration acceptance
and reactivation are capped at 10% (`fminf(vertexScore, 0.1)`), throttling exploration so
cost dominates node retention. Pruning driven purely by cost.

## Quick comparison
| Variant | Explore | Cost-aware | Pruning | Spatial hash |
|---|---|---|---|---|
| KPAX | Syclop | — | — | — |
| PruneKPAX | ✔ | goal-dir | goal-progress | ✔ |
| KinoPaxPlus | weak | ✔ | dormancy/cost | — |
| KinoPaxSTARNoPrune | ✔ | ✔ | — | ✔ |
| KinoPaxSTARNoPruneNoSpatialHash | ✔ | ✔ | — | — |
| KinoPaxSTAR | ✔ | ✔ | goal-progress | ✔ |
| KinoPaxSTARcostprune | capped 10% | ✔ | cost | ✔ |
