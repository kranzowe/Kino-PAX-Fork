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

### KPAXCap  *(new)*
Stock KPAX plus one field: `h_syclopCap_`, a multiplier in (0,1] on the Syclop region score,
applied at **both** acceptance points — `curand < cap * vertexScores[r] || !activeSubVertices[sub]`
in propagate, and `curand <= cap * vertexScores[r] + fAccept` at dormant-node reactivation.
`fAccept` is **not** scaled (it is the additive floor dormant nodes rely on; multiplying it by a
small cap would switch revival off rather than throttle it), and the R2 seeding disjunct is **not**
capped either — it stays a free pass. `cap = 1.0`, the default, is bit-identical to KPAX and adds no
RNG draw, so it is a genuine no-op. Same contract as `KinoPaxSTARTrue::h_syclopCap_`.

**Why it exists.** `KinoPaxSTARCleanCost` at `w = 1, cap = 1` is *not* KPAX — it is strictly
stricter, because it changes two things at once: it applies the cap, **and** it moved the acceptance
decision past the kernel boundary. The second one is not neutral. `computeVertexScores_kernel`
divides by `(1 + counterArray²)` with `counterArray` cumulative over the whole run, so the regions
the frontier currently occupies take a large counter jump *within* an iteration; KPAX admits that
batch on the pre-jump score, while CleanCost's accept kernel runs after `graph_.updateVertices()`
and judges it on the post-jump, quadratically-crushed score. (`updateSampleAcceptance_kernel`'s
`vertexScores[r] = 1.0` free pass for regions with `validCounterArray[r] == 0` is likewise already
spent by gate time.) KPAXCap changes only the cap, still deciding inside propagate on pre-jump
scores — so the triple KPAX / KPAXCap / CleanCost-at-`w=1` separates "the cap did this" from "the
kernel boundary did this".

Kernels are prefixed `KPAXCap_*`. That is load-bearing, not cosmetic: KPAX's kernels are
*unprefixed* global symbols and `CUDA_SEPARABLE_COMPILATION` is ON, so an unprefixed copy is a
duplicate-symbol link error. `originalKPAX.cu` already sets the precedent with its `original_*`
prefix.

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
### KinoPaxSTARNoGoalBias  *(renamed from KinoPaxSTARNoPrune)*
The old name was ambiguous: "prune" was being used for two unrelated mechanisms — the
goal-biased admission gate (`_goalProgressPrune_kernel`, really an extra acceptance criterion)
and KinoPaxPlus's retroactive cost pruning. `NoPrune` meant the former. Goal-bias logic lives
**only** in `KinoPaxSTAR`; every other STAR variant has none, and the rename makes that legible.
The CSV label string `"KinoPaxSTARNoPrune"` is retained in the older large/delta benchmarks so
their historical data still loads.

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

### KinoPaxSTARWeightedCost  *(new)*
KinoPaxSTARcostprune with a **weighted-sum** acceptance rule instead of a multiplicative one.
costprune combines the two probabilities as `costProb * syclop`, which collapses to zero in
exactly the cells that hold a narrow passage: `Graph.cu:249` drives the Syclop score down
quartically where the valid-sample fraction is low, and multiplying by `costProb <= 1` destroys
even the additive `fAccept` term KPAX relies on to keep resampling there. This variant uses

    P_combined = min(1, w*P_syclop + (1-w)*P_cost + P_floor)

at **both** acceptance points — the insertion gate `_costPrune_kernel` and the Part-B reactivation
branch — with `P_syclop = vertexScores[xR1] + fAccept` (the full KPAX rule) and `P_floor` fixed at
`EPSILON`, matching the floor already baked into the Syclop score itself. One knob: `h_costWeight_`
(w) = 1 reproduces KPAX's acceptance, 0 is pure cost-greedy. `h_acceptCap_` survives but governs
only the propagate-time dual-track acceptance, not either weighted decision.

`P_cost` is `costProbExp` (`helper.cuh`), **not** `costKeepProb`: `exp(-k*(cost-m)/(mean-m))` is
exactly 1 at the region min *and* has a real gradient across the whole range, where
`min(1,(mean/cost)^k)` is pinned at 1 for every cost at or below the mean (so half of each region
gets no discrimination) and never reaches 0. It carries no floor of its own — `P_floor` is added
once, in `weightedAccept`. `h_costPruneExp_` (k) stays the sharpness knob. The `cost <= m` early
return in the gate is load-bearing here: at the region min `P_cost = 1` but `P_combined` is still
below 1 whenever `P_syclop < 1`, so without the exemption a region best could be rejected.

Note `h_fAccept_` is computed *before* the gate in this variant (the gate needs it), so
`treeAddSize` reflects the pre-gate frontier size — marginally smaller `fAccept` than costprune's.
Carries **no** retroactive pruning (an earlier revision of this file claimed otherwise); the
cost-guarded version is `KinoPaxSTARTrueWeightedCost`.

**Superseded by `KinoPaxSTARCleanCost`.** The two acceptance points here compose as an AND, so the
effective admission probability for a non-best candidate is `[≤ h_acceptCap_, or fresh-subregion] ×
weightedAccept(...)`, not `weightedAccept(...)`. Retained so its historical benchmark data stays
loadable and as the A-side of the fold.

### KinoPaxSTARCleanCost  *(new)*
`KinoPaxSTARWeightedCost` folded down to **one** acceptance decision.

WeightedCost ran two, and they compose as an AND for every non-best candidate: a propagate-time
filter (`isBest || rand < min(vertexScore, h_acceptCap_) || !activeSubVertices[sub]`, cap = 0.1,
no `fAccept`) and then the weighted gate. Three consequences — `h_acceptCap_` sat silently upstream
of `w`, so `w` was never the single knob it was documented as; the R2 seeding "free pass" was not
free, since a fresh-sub-region node still faced the weighted roll; and the tree grew roughly an
order of magnitude slower than the weighted rule alone implies.

**Why the gate is the only valid place for the decision.** Two inputs are invalid inside the
propagate kernel and become valid at the next kernel boundary. Region cost statistics are
*mid-flight*: `minCostsR1` / `sumCostsR1` / `cntCostsR1` are being updated by atomics from the very
threads that would read them, so `costProbExp` computed there would use a partial mean over
whichever threads landed first — two identical-cost nodes in one region would draw different
`P_cost` from scheduling alone. (Tolerable for `isBest`, where being wrong only keeps an extra node;
not tolerable for the distribution the weighted rule imposes.) And `vertexScores` are one iteration
*stale* there, because `graph_.updateVertices()` runs between the two kernels. `fAccept` is a third:
it is computed from `h_frontierNextSize_`, which does not exist until propagate has finished.

So propagate is a pure **candidate producer** — mark every collision-free sample, record its cost,
region, and R2 freshness, draw no random numbers — and the renamed `_accept_kernel` (it was never a
prune; nothing is in the tree yet) makes the single decision:

    P = cap * min(1, w*P_syclop + (1-w)*P_cost + P_floor)

applied identically at the gate and at Part-B reactivation, with `P_syclop = vertexScore + fAccept`.
Two exemptions precede it: region-best candidates (`cost <= m`, load-bearing — at the min `P_cost`
is 1 but `P_combined` is still below 1 whenever `P_syclop < 1` or `cap < 1`), and candidates that
claimed a virgin R2 sub-region, which is now an actual free pass. The freshness flag has to be
recorded in propagate because by gate time every sub-region touched this iteration is already
marked active; the read-then-set order is kept, so a whole launch landing in one virgin sub-region
all get the pass, exactly as in KPAX.

Three orthogonal knobs: `h_costWeight_` (w), `h_costPruneExp_` (k), and `h_acceptCapMul_` (cap in
(0,1], default 1.0) — a flat multiplier on the final probability that replaces `h_acceptCap_`,
downstream of `w` rather than upstream of it. Note `cap` multiplies `P_floor` too, so the effective
floor is `EPSILON*cap`; under WeightedCost it was ~`0.1*EPSILON` because the propagate stage
throttled it. Also drops `d_maxCostsR1_` (`costProbExp` normalizes by `mean - min` and never reads a
max, so the per-sample `atomicMaxFloat` was pure overhead) and `d_pruned_` (nothing ever wrote it).
Carries no retroactive pruning.

Because the propagate-time throttle is gone, this admits far more per iteration than WeightedCost at
the same `w` — expect to retune, with `cap` as the compensating knob.

**Two normalization fixes (current).** Both of the following were diagnosed by walking the
formula, and both flattened acceptance into "cap × a constant":

*1. The Syclop floor was larger than the score it floored.* `updateSampleAcceptance_kernel` sets
`vertexScore = floor + score/total`, and the `score/total` shares sum to exactly 1 across active
regions — so the mean share is `1/N_active`, 3.7e-5 at 27k regions, against a fixed
`EPSILON = 1e-2`. Because the shares sum to 1, **at most `1/EPSILON = 100` regions can rise above
the floor at any grid size**; everything else sits at exactly 0.01, and refining the grid makes it
worse (same budget of 1.0, split more ways). `Graph::h_dynamicScoreFloor_` replaces it with
`1/N_active` — the mean share itself — so an average region gets exactly 2× the floor at any
discretization.

**Opt-in, not global.** `KPAXCap`, `KinoPaxSTARTrue` and `KinoPaxSTARCleanCost` set
`graph_.h_dynamicScoreFloor_ = true`; **`KPAX` deliberately does not**, so it remains an unmodified
historical baseline. Every other `Graph`-based variant also keeps the legacy floor. `KinoPaxPlus`
is unaffected either way — it uses `KinoPaxPlusRegions`, not `Graph`, and never reads
`vertexScores`. The `count_if` only runs in dynamic mode, so legacy planners pay nothing.

*2. The cost term was normalized per-region, so `x ≈ 1` everywhere by construction.* `costProbExp`
divides by the region's own spread `(mean_r − m_r)`, which pins the typical candidate at x ≈ 1 in
*every* region simultaneously — that is what dividing by the mean does — so `exp(−k·x) ≈ exp(−k)`
was uniform across the grid and raising k just multiplied the same near-zero everywhere. It also
made k mean different things in different places: a corridor region whose costs span 2 units
punished a 2-unit excess exactly as hard as an open region punished a 30-unit one.
`costProbExpGlobal` keeps the region's own minimum as the **reference** — preserving the bias
toward each region's cheapest node, which is what stops the search collapsing onto the root where
all the globally-cheapest nodes live — but divides by a **global scale**
`D_global = globalMeanCost − globalMinCost`. Both terms are in cost units, so it is COST_MODE-safe:
no heuristic, no mixing distance with control effort.

`h_probFloor_` is now **0** in CleanCost. `vertexScore` already carries the Graph floor, so the
second additive EPSILON was duplication — and together the pair contributed an unconditional
`0.9·0.01 + 0.01 = 0.019` that swamped the cost term outright.

**Where `cap` stands now.** The `cap ≈ 1/h_activeBlockSize_` derivation still holds structurally —
after the fold each frontier node offers `repeat · blockSize` candidates to one rule, so the
per-node branching factor is `b = repeat · blockSize · ν · cap · p̄` and holding `b ≈ 1` gives
`cap ≈ 1/(repeat · blockSize · ν · p̄)`. But it was *calibrated* against a `p̄` inflated by the two
floors. With both fixed, the cost-independent part falls from ~0.019 to ~0.002, so the implied cap
rises by roughly the same 8× — **`cap = 1.0` may now be correct**, and the sweep re-opens the axis
upward to find out.

**R2 seeding free pass is switchable and now DEFAULTS OFF** (`h_r2SeedAccept_ = false`). Both arms were measured head to head and off is the permanent condition: admission is steered only by the Syclop score and the cost term. A candidate that
claimed a virgin R2 sub-region is normally admitted unconditionally, bypassing the weighted roll —
this is KPAX's main coverage drive, and in CleanCost it is a genuine free pass (in
`KinoPaxSTARWeightedCost` it only cleared the propagate-time filter and then still faced the
weighted roll). Setting it `false` makes such a candidate take the same roll as everything else,
i.e. the `KinoPaxSTARnoseed` condition (pSeed = 0), so the on/off pair measures how much of the
exploration is seeding rather than the Syclop score.

Only *admission* changes: propagate still marks `activeSubVertices` unconditionally, so
`r2_coverage_pct` stays a valid, comparable diagnostic in both arms — the same contract the other
noseed variants keep. A bool rather than `KinoPaxSTARcostprunenoseed`'s float `h_pSeed_`; widen it
to a probability if the annealed middle ground is wanted.

**`EPSILON` is overloaded — do not redefine it.** It does three unrelated jobs:

| use | site | status |
|---|---|---|
| Syclop floor, `EPSILON + score/total` | `Graph.cu` `updateSampleAcceptance_kernel` | **replaced** by `h_scoreFloor_` (opt-in) |
| Laplace smoothing, `(EPSILON + valid)/(EPSILON + total)` | `Graph.cu` `computeVertexScores_kernel` | leave — only guards 0/0 |
| fAccept scale, `(h_itr_ * EPSILON) * treeAddSize^5` | **15 planners** | leave — redefining EPSILON silently rescales reactivation everywhere |

**Diagnostics.** The sweep logs `score_floor` and `cost_scale` per iteration. `score_floor` sits
flat at 0.01 for KPAX and decays as `1/N_active` for the opted-in planners — the direct evidence
the floor fix is live. `cost_scale` is `D_global`; compare it against the per-region spreads that
used to be the denominator to pick the next k range, since `x_new = x_old · (mean_r − m_r)/D_global`
and the right k depends entirely on that ratio.

**Discretization note for the sweeps.** `NUM_R1_REGIONS = W_R1^3 * V_R1^3` has no C term, and Model 1
sets `C_DIM 0`, so `getRegion` / `getSubRegion` skip the C dimension entirely — `C_R1_LENGTH` is a
no-op and stays at 1. Control-side refinement rides on `V_R1`. That makes the sweep's `fine`
(`W_R1` 10→20) and `fine_control` (`V_R1` 3→6) a controlled pair: **identical 216,000 region count,
refined in workspace vs in velocity.**

### KinoPaxSTARCOMBO  *(new)*
`KinoPaxSTARCleanCost` with the acceptance **cap replaced by a growth controller**, and the weighted
sum replaced by three sigmoids over globally-normalized metrics.

The propagate/gate structure is copied verbatim — propagate is a pure candidate producer, one
acceptance rule runs in `_accept_kernel` after `graph_.updateVertices()`. Two things change.

**1. What the probability is a function of.** `weightedAccept(w, vertexScore + fAccept,
costProbExpGlobal, floor)` becomes `comboShape` (`helper.cuh`):

**TWO shapes, not one** — because one rule was answering two different questions:

```
u  = treeSize / MAX_TREE_SIZE            v = u^(ln0.5/ln mid)      wCov = (1-v)^g,  wCst = v^g

shape_accept = ( σ(-kAccCov·d1)·wCov + σ(-kAccCst·d3)·wCst ) / (wCov + wCst)     g = g1
shape_fanout = ( σ(-kFanCov·d1)·wCov + σ(-kFanCst·d3)·wCst ) / (wCov + wCst)     g = g2

    d1 = (r1Coverage − exploredMeanCoverage) / exploredMeanCoverage   prefer thin regions
    d3 = (nodeCost   − r1MeanCost)           / costScale              prefer cheap nodes
```

Both in (0,1), both exactly **0.5** at the neutral point for every `g`, `mid` and `u`. `shape_accept`
gates admission; `shape_fanout` sizes `rep`. **Cost belongs in acceptance and is counter-productive
in fan-out**: cost is cumulative root-to-node, so "cheap" means *shallow*, and weighting fan-out by
it pours propagation around the root — a density mechanism where KPAX's novelty rule is a reach
mechanism. That is why COMBO grew a bigger tree than CleanCost while reaching a first solution later.

Each shape **blends coverage into cost as the run progresses**: `mid` sets where the crossover sits,
`g` how sharply. `g` alone cannot move the crossover (`(1-u)^g` and `u^g` are equal at `u = 0.5` for
every `g`), hence the separate midpoint. Normalising by `wCov + wCst` rather than by a constant is
load-bearing: at `g = 1` the weights already partition unity, so a constant divisor of 2 would return
0.25 at neutral, and at `g ≠ 1` they do not sum to 1 at all, so a constant divisor would let `g`
rescale the shape instead of only reshaping the transition.

**The collision term is gone** (may return later); `h_globalCollisionFrac_` survives as ν's source.

**How a smooth fan-out shape reproduces KPAX's sparsity.** A node is favoured — 15 blocks instead of
1 — when its fan-out score exceeds `μ + N·σ` over the score distribution of the **whole realised
frontier**. `kFan` controls σ: at low gain the shape crowds around 0.5, σ is small, and the threshold
sits just above the mean of a *right-skewed* distribution, so it favours the **majority**. At high
gain the sigmoid degenerates to a step, the shape goes bimodal `{≈0, ≈1}`, σ is large, and the
threshold lands squarely between the two modes — KPAX's 15/1 with an **adaptive** threshold
(relative to the frontier's own spread, recomputed each iteration) rather than a hardcoded `< 10`.
So `kFan` is the headline tuning axis alongside `N`, *low* gain is the failure mode, and `kFan = 0`
is the uniform-fan-out control arm (σ = 0, degenerate branch, every frontier node the same count).

Every `d` is signed so `d > 0` means unfavourable, and every one is divided by a **global**
reference. Both halves of that are load-bearing. Raw deltas have no usable range: with
`NUM_R2_PER_R1 = 64` a one-sub-region coverage difference is 0.0156 and `σ(0.0156) = 0.4961`, so T1
would be a constant, while `(cost − mean)` is O(10²) in cost units so T3 would saturate to a step. A
*local* scale is worse still — dividing by the region's own spread pins the typical candidate at
`d ≈ 1` in every region by construction, which is what made the old `k` knob inert (see
`costProbExpGlobal`).

**`k` is a gain in the argument, not an exponent.** `σ(x)^k` moves the midpoint to `2^-k` and its
slope actually *falls* past `k = 2`; `σ(k·x)` holds the midpoint at 0.5 for every `k`. That is what
makes the neutral value exact for all `k` rather than only for `k = 1`. `k_i = 0` pins term
*i* at 0.5 — an exact ablation switch, which is how the sweep isolates the three terms.

`h_costWeight_`, `h_costPruneExp_`, `h_probFloor_` are gone. So is `h_fAccept_`: reactivation
pressure is now set explicitly by `h_reactFrac_` rather than by a decaying nudge on a term that no
longer exists.

**2. How it is scaled — `h_acceptCapMul_` is gone.** A cap is a constant, but the probability that
hits a given growth rate is not:

```
pTargetAccept = (wantThisIter − exempt) / ((candidates − exempt) · meanShape)
```

`candidates` falls over a run as the tree buffer fills and the fan-out is forced down, so the
required value **rises ~5×**. That is why every earlier variant needed a hand-swept cap and why no
single value was ever right at both ends — the empirical 0.03 is this expression evaluated near the
end of a run. Here it is computed from measured quantities every iteration: feedforward and
deadbeat, no gain to tune.

**Two budget scalars, not one.** The gate judges ~10⁶ candidates; Part B judges the whole tree.
CleanCost shares one scalar only because its P is ~1e-4, so reactivation is a trickle; at the P this
planner needs, a shared scalar would reactivate more nodes per iteration than the entire growth
target and the frontier would run away. The *shape* is shared — that is the CleanCost invariant that
matters — the budget cannot be.

**Fan-out replaces the binary 15/1 at both sites, and it is decided in `propagateFrontier`, not in
the update kernel.** Every branch that puts a node in the frontier records the fan-out score it was
admitted with into a tree-indexed array. Then, immediately after `findInd` has compacted the
frontier, the planner reduces μ and σ over it, counts `nFav` above `μ + N·σ`, and solves

```
Σrep = F + (repHi − 1)·nFav  ≤  blockCeiling
blockCeiling = min(selectivity·want/ν, 0.8·remaining) / activeBlockSize
```

for `repHi`, capped at `h_repeatMax_` and floored — never rounded — at 1.

**The placement is the whole point.** At that instant the compacted list *is* the whole proposed
frontier: Part A admissions, min-cost exemptions, Part B region-bests, Part B reactivations, repair
arm. So `F` and `nFav` are **counted**, not estimated. Sized in the update kernel they could not be
— Part B's reactivation rolls have not happened yet there, so `fNext` was an estimate and `nFav` a
prediction from a fraction measured over *pre-gate candidates against the previous iteration's
threshold*. Both ran small early in a run, `repHi` pinned at `h_repeatMax_`, and `Σrep` came
uncoupled from the budget it was solved against — which dropped propagate onto the slow kernel2 path
sporadically in early iterations. Kernel2 is still forced once `32·F > remaining` (`rep ≥ 1` is a
correctness floor and the region-best reactivation is unconditional, so `F ≥ nActive`) — roughly
**59%** of the tree in the sweep config — but that is now the only route to it.

It is also **the only writer** of `activeFrontierRepeatCount`, running over exactly the compacted
frontier. `rep ≥ 1` therefore holds by construction rather than by clamp, and two traps disappear
with it: no frontier node can be left blockless (which stranded its bit forever, since kernel1 clears
the bit from the expanding block), and no node outside the frontier can hold a count (which made
`repeatInd` emit a slice no thread writes, fathering phantom-parented nodes at cost 0). Goal nodes no
longer need their count cleared, so Part A's "must stay last" ordering constraint is gone too.

**Knobs.** Shape: `h_kAcc*` / `h_kFan*` (default 4.0, 0 = ablate). Fan-out: `h_fanSigmaN_` (1.5).
Budget: `h_selectivity_` (120 — the measured candidates-per-admission of a well-tuned CleanCost run),
`h_reactFrac_` (0.1), `h_growthIters_` / `h_growthExp_` (schedule; `exp = 1` is linear), and the
limits `h_repeatMax_` (15) / `h_pMax_` (0.5). **None of them is a scale factor on a probability** —
each describes what you want or is a safety limit.

**R2 seeding is removed outright**, not switched off: `h_r2SeedAccept_`, `d_frontierNextFresh_` and
the accept kernel's second exemption are all gone. Propagate still *marks* `activeSubVertices`, so
`r2_coverage_pct` and the new `d_regionCoverage_` stay valid; a virgin sub-region simply no longer
buys admission. The `ACC_SEED` counter slot is retained and permanently 0 so the CSV schema stays
comparable with CleanCost's.

**The min-cost exemption is kept** as an unconditional free pass at both acceptance points —
optimality convergence depends on every region's best staying in the frontier. Exempt nodes carry a
**real** fan-out score, computed by the same function as everyone else's, and compete on the same
`μ + N·σ` threshold. Cost is what got them in the door and must not also buy them propagation: cost
is cumulative root-to-node, so "cheap" means *shallow*, and there are ~4.5e3 exemptions per
iteration plus one region-best per explored region.

**Two bugs fixed relative to CleanCost, both found by auditing the data flow rather than by a run.**
`activeFrontierRepeatCount` is zeroed wholesale each iteration and Part B's reactivation is gated on
`frontier[treeIdx] == 0`, so a node that ends an iteration with `frontier == true` and count 0 is
never expanded (kernel1 clears the bit only from the expanding block) and never re-counted — the bit
sticks true forever, inflating `h_frontierSize_`, and the same guard rejects it on every future
iteration. COMBO adds an explicit repair arm — now unreachable on the kernel1 path, since the
fan-out assignment covers the compacted frontier exhaustively, but kept because kernel2 clears
`frontier[tid]` by *thread* index rather than by the node it expanded. Separately, `resetPlanner`
now zeroes
`d_acceptCounts_` / `h_acceptCounts_`, which CleanCost never did — a planner object reused across
runs carried the previous run's final counts into iteration 1, and here those counts feed the
controller.

**Additive `Graph` change, shared by every planner:** `d_regionCoverage_` materializes the per-region
coverage `computeVertexScores_kernel` already computed as a local and discarded, and `h_nActive_`
keeps the count the dynamic score floor already made. The score still consumes the same local the
same way, so every existing planner is bit-identical. `IsActiveRegion` moved from file-local in
`Graph.cu` to `Graph.cuh` so planners reduce over the same active set rather than defining a second
copy.

**Known exposure worth watching.** Dropping `vertexScores` from acceptance also drops Syclop's
`1/(1 + counterArray²)`, the only term that penalized an over-sampled region. Coverage (T1) is the
intended replacement — but coverage is cumulative and monotone toward 1.0, so once it saturates T1
goes constant and that penalty is gone with it. `explored_mean_coverage` is logged per iteration
precisely to show when. The lever if it matters is config, not code: `W_R2_LENGTH`/`V_R2_LENGTH`
2→3 gives 729 sub-regions per R1 instead of 64.

Swept by `examples/gpu/kinopaxstar_combo_tuning_sweep.cu` (profile × gain, 13 points; the
all-gains-zero `none` point is the control — shape is a constant, so it measures the controller with no
metric steering at all). Opts into the dynamic score floor. Carries no retroactive pruning.

### KinoPaxSTARTrue  *(new)*
`KinoPaxSTARNoGoalBias` plus **cost-guarded** retroactive pruning. A node is pruned when it was
admitted *because* it was its region's minimum and no longer is; nodes the Syclop exploration roll
admitted are never touched. `h_ancestorPrune_` is an on/off toggle: 0 = off (reproduces
`KinoPaxSTARNoGoalBias` exactly), nonzero = stale-best. `h_ancestorTol_` (0) matches KinoPaxPlus.

**`h_syclopCap_`** *(new)*: a multiplier in (0,1] on the Syclop region score, applied at both
acceptance points — `curand < cap * vertexScores[r]` in propagate, and
`curand <= cap * vertexScores[r] + fAccept` at Part-B reactivation. `fAccept` is deliberately **not**
scaled: it is KPAX's additive reactivation floor, and multiplying it by a small cap would switch
dormant-node revival off rather than throttle it. `cap = 1.0` (the default) reproduces the previous
behaviour exactly and adds no RNG draw either way, so it is a genuine no-op at the default.

**The ancestor-chain mode (formerly 2) has been removed.** The guard returns before the recurrence
for any Syclop-admitted node, so `ancestorBad` was never written for one and stayed `false` forever —
and `ancestorBad[i] = selfBad(i) || ancestorBad[parent(i)]` then read "never asked" as "clean". The
chain silently truncated at the first explorer ancestor, and since explorers are the majority of a
STAR tree it already degenerated toward stale-best in practice. Call sites still passing 2 get
stale-best.

**`h_dormancyThreshold_` (5) no longer has any effect.** With the chain gone, `pruned[]` is only ever
set under `selfBad`, and at `h_ancestorTol_ = 0` that is exactly `!isBest`. `isBest` is *monotone* —
`minCostsR1` is filled once with `MAX_FLOAT` and thereafter only lowered by `atomicMinFloat`, and
`treeSampleCosts[i]` is written once at insertion (no rewiring) — so once pruned, a node can never be
region-best again, and the dormancy branch's `pruned && isBest` can never hold. `inactiveIterations`
is incremented only inside that branch, so the amnesty branch never fires either. Both branches only
did work under the chain, which could tombstone a node that was *still* region-best because an
ancestor went bad. They are left in place to preserve the KinoPaxPlus lineage. The field is retained
so existing benchmark call sites keep compiling.

**Why the guard exists.** Its predecessor, `KinoPaxSTARNoPruneAncestor`, applied
`cost > minCostsR1[r]` to *every* tree node. Syclop-admitted nodes are non-minimum by
construction — they were admitted despite failing the `isBest` test — so all of them were
tombstoned on the first pruning pass, and since Part B returns early on `pruned[]` they never
reactivated. The only escape, the dormancy branch, rehabilitates a node only once it has *become*
region-best, which an explorer never does. The entire exploration population froze. The rule is
faithful to KinoPaxPlus; the population was wrong — KinoPaxPlus's tree is almost entirely
min-cost nodes because `pruningFrontier_kernel` hard-rejects `cost > minCostsR1` at insertion, so
the same rule removes almost nothing there. That variant is retired.

The admission reason is recorded at propagate time as `isBest && !acceptedByExploration` and
carried into the tree alongside `treeXR1s`; both flags are already computed before the acceptance
`if`, so the recording touches no RNG state and mode 0 is bit-identical to the base class.

### KinoPaxSTARTrueWeightedCost  *(new)*
`KinoPaxSTARWeightedCost` plus the same cost-guarded pruning and the same three modes. The
weighted-sum acceptance and `costProbExp` are unchanged; benchmarks run it at w = 0.9, k = 1.

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
each point `cap{100*cap}` plus `_exp{100*exp}` — e.g. `KinoPaxSTARcostprune_cap33_exp50`. Two
label conventions are in the data: `kinopaxstar_cost_benchmark.cu` swept a hand-picked 5-point
list and omitted `_exp` at `exp == 1.0` (`..._cap40`, `..._cap0_exp75`), while the newer
`kinopaxstar_cost_tuning_sweep.cu` runs the full
`cap {0, 0.33, 0.66, 1.0} x exp {0.1, 0.5, 1.0} x floor {0.1}` grid (the 0 and 0.2 floor
columns were dropped once measured) and always spells
every knob out (`..._cap33_exp50_floor10`), so the two data sets never collide. The floor is
worth sweeping because `costKeepProb` ends in `fmaxf(p, floor)`: any non-zero floor grants every
non-best node that much survival chance regardless of cost, so only `floor = 0` lets the gate
approach region-best-only retention. The tuning sweep
also varies the cost metric, which is a compile-time property of the binary (`COST_MODE` in
`helper.cuh`) and therefore rides in the delta token instead: `large_effort` (control effort)
vs `large_length` (workspace path length). Cost dominates node retention.
`h_reactivationBlend_` switches the reactivation blend: 0 = `costProb * syclop` (the original
intersection), 1 = `fmaxf(costProb, syclop)` (union — cost-promising OR exploration-promising,
matching the rule the propagate kernels already use for admission). Mode 1 restores the additive
`fAccept` floor and **must be paired with `h_costPruneFloor_ = 0`**: `costKeepProb` ends in
`fmaxf(p, floor)`, so under the union a non-zero floor becomes a blanket reactivation probability
for every dormant node in the tree.

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

### CountingStars  *(new)*

Derived from KinoPaxSTARCOMBO. **v1** (branch `CountingStarsAlgorithm`) replaced COMBO's acceptance
probability with per-region counts. **v2** (branch `CountingStarsV2`, same class name, evolved in
place) replaces those counts with **one global node budget, filled in priority order**. COMBO stays
as the comparison arm; v1 stays on its own branch, which does mean v1 and v2 cannot appear in one
figure without merging the two.

**Why the probability had to go (v1's argument, still load-bearing).** COMBO admitted with
`P = min(pMax, shape · pTargetAccept)`, where `pTargetAccept` was solved so the *expected* admission
count hit a growth target. Two properties follow, and together they make the rule unable to do the
job it existed for:

1. The shape is a normalised blend of sigmoids — neutral 0.5, ceiling 1.0 — so the best candidate is
   at most **2×** as likely to enter as an average one.
2. `pTargetAccept = (want − exempt)/(rolled · meanShape)` divides by the *measured* mean. Sharpen the
   shape, the mean falls, `pTarget` rises to compensate, and the gain is handed straight back.

Acceptance was a **reallocation** mechanism at a fixed total. It could not concentrate, so nodes came
out dense everywhere instead of sparse where it mattered.

**Why the counts had to go too.** v1 admitted by per-region quotas and the global frontier size `F`
was whatever those happened to produce. That is backwards: **GPU throughput is a function of frontier
size, so frontier size should be the input.** v2 inverts it — `goal_frontier_size` **B** is the
primitive, tunable to whatever the GPU is fast at, and the doors fill it. The budget is met *by
construction*, not tracked, which is the same discipline that already makes the block ceiling work
(`F` and the frontier's block demand are both counted before the launch).

This also retires the pattern that has failed three times in this line: **steering a global quantity
through per-region knobs with feedback.** COMBO fed back its fan-out threshold; COMBO fed back its
surplus `repHi`; a throughput controller over v1's counts would have been the third.

**Four doors, one budget, priority order.**

| # | Door | Rule | Budget share |
|---|---|---|---|
| 1 | **Optimal** | `distance == 0`, i.e. `cost <= minCostsR1[r]` | **uncapped**, first claim every iteration |
| 2 | **Freshest** | region ordinality below the cutoff (+ a boundary roll) | `explore_frac ·(B − optimalCount)` |
| 3 | **Guarantee** | `bestNodeIdxPerR1[r]` for every active `r` with `!regionCovered[r]` | whatever doors 1–2 left |
| 4 | **Draw** | uniform over the rest of the tree | `p = (B − admitted − guaranteed)/treeSize` |

`distance = (cost − minCostsR1[r]) / costScale`, with CleanCost's global scale
`costScale = globalMeanCost − globalMinCost`. The optimal door having **first claim every iteration**
is a stronger optimality guarantee than v1's region-best *reactivation*, which only restored a
region's best after it had already been passed over.

Optimal is uncapped and safe while **B > NUM_R1_REGIONS**, since `NUM_R1_REGIONS` bounds how many
nodes can be a region best in one iteration. Below that, B is a **soft target** — deliberately, and
the sweep visits two such points to measure how much of the frontier the top door alone accounts for.

**No sort, and not for performance reasons.** `distance 0` is a *threshold*, not an order.
Top-X-freshest *is* an order, but ordinality is a small non-negative integer, so a **histogram plus
an exclusive scan** gives the exact cutoff in two O(n) atomic passes. (A sort would also have been
affordable: `thrust::sort_by_key` dispatches to CUB radix sort at ~1–2 G keys/s on Pascal, so
10⁵–3×10⁵ candidates is 0.05–0.3 ms against a ~15 ms iteration. It is absent because it is
unnecessary, not because it is slow.) The **boundary roll** on the cutoff bucket is what makes the
count exact rather than approximately right: the X-th freshest node almost never falls on a bucket
edge, and taking the whole boundary bucket would overshoot by up to one bucket's width.

**Ordinality is per-region, not per-candidate.** Every candidate in region `r` shares
`regionNodeCount[r]`, so "freshest" means "from the least-populated region" — the novelty signal we
want, at the cost of a single read and no per-candidate counter. Ties break arbitrarily, which is
exactly what the boundary roll resolves.

**Two accept passes, and the split is what makes the budget exact.** The histogram must be complete
before the cutoff is known, and the cutoff before anything is admitted. Pass 1 measures and stamps no
door; pass 2 decides and is the only door writer, so the counters cannot double-count.

**Fan-out is a split, not a ramp.** `blockBudget = maxBlocks · B`; optimal nodes take `maxBlocks` each
off the top and everyone else divides the rest evenly, floored, `>= 1`. It is deliberately
*non-binding* while the frontier lands at or under B (the divisor is then `B − optimalCount` and
everyone gets `maxBlocks`) and bites on an **overshoot**, where the optimal door keeps its full boost.
v1's geometric ramp `max(maxBlocks >> (ordinal/halfLife), 1)` is **gone with the explore door that
indexed it** — ordinality is now a *selection* signal, and running it as a *weighting* signal too
would double-count the same fact. The buffer bound
`blockCeiling = 0.8·remaining/activeBlockSize` is a **separate constraint** and both must hold;
`propagateFrontier` still scales the *boost* and never the `rep ≥ 1` floor, which keeps it the single
writer of `activeFrontierRepeatCount`.

**The R2 door is gone; the R2 marking survives.** Novelty is ordinality now, so no door reads a
sub-cell. The claim is kept — as a **read-then-CAS**, one increment per cell ever — purely so
`r2_coverage_pct` stays comparable with the KPAX-family baselines at O(1); the alternative is a
`thrust::count` over `d_activeSubVertices_` every iteration, 2.1M elements at the coarse delta and
37.9M at `tiny`.

**It carries a corrected R2 mapping, and only it does.** `getRegion` encodes
`r1 = wRegion·C_R1_LENGTH^C_DIM·V_R1_LENGTH^V_DIM + aRegion·V_R1_LENGTH^V_DIM + vRegion`, but
`Graph.cu`'s `initializeRegions_kernel` decodes it with the **digit order reversed** *and* with
hardcoded exponents `C_R1_LENGTH²` / `V_R1_LENGTH¹` where the encode uses `C_DIM` / `V_DIM`. The
collapse factor is `C_R1_LENGTH^(C_DIM−2) · V_R1_LENGTH^(V_DIM−1)` — **8× at the checked-in config,
and worse at a finer discretisation**. Since `d_minValueInRegion_` is consumed only by
`getSubRegion`, R1 statistics are fine and every R2 identity is wrong. It no longer decides anything
here, but the coverage metric is only comparable if the cells being counted are the right ones, so
the corrected corner table stays. `Graph.cu` is left alone so existing baselines stay comparable,
which does mean COMBO's coverage delta and KPAX's seeding door keep reading the scrambled signal.
`scripts/check_region_math.py` proves the corrected decode is a bijection and measures the shared
one's collapse.

**Knobs.** `h_goalFrontierSize_` (10000), `h_exploreFrac_` (0.1) and `h_maxBlocks_` (16) — all
three swept. `maxBlocks` is **not** a restatement of `B`: while the fan-out split is non-binding
every frontier node receives `32 · maxBlocks` propagations, so `B` sets the frontier's *size* and
`maxBlocks` sets propagations *per node*.

**Cost acceptance is permanent.** A `h_costAccept_` toggle briefly existed to test whether the two
cost-driven doors — the **OPTIMAL** door in accept pass 2 and the **GUARANTEE** in Part B — were
what held time-to-first-solution back. It has been removed: those doors are what makes the search
converge on cost at all, and without them nothing preferentially expands cheap nodes. Both always
run.

**Removed in v2:** `h_exploreCount0_/1_`, `h_costCount0_/1_`, `h_reactCount0_/1_`,
`countingStarsRamp()`, `h_fanHalfLife_` and the geometric ramp, `d_novelCounts_`, `d_candNovel_`, and
the per-region quota logic in the accept kernel. `d_regionNodeCount_` survives and becomes the
ordinality source. **Removed in v1, still absent:** the whole shape apparatus (`comboShape2`,
`comboDeltas`, `comboBlendWeights`, μ+N·σ, every `h_kAcc*`/`h_kFan*`/`h_blend*`), both budget
scalars, `h_selectivity_`, the growth schedule, and every statistic that existed only to feed them.

## Quick comparison
| Variant | Explore | Seeding (pSeed) | Cost-aware | Pruning | Spatial hash |
|---|---|---|---|---|---|
| KPAX | Syclop | 1 | — | — | — |
| KPAXCap | capped (`h_syclopCap_`) | 1 | — | — | — |
| PruneKPAX | ✔ | 1 | goal-dir | goal-progress | ✔ |
| KinoPaxPlus | weak | n/a | ✔ | dormancy/cost | — |
| KinoPaxSTARNoGoalBias | ✔ | 1 | ✔ | — | ✔ |
| KinoPaxSTARNoPruneNoSpatialHash | ✔ | 1 | ✔ | — | — |
| KinoPaxSTAR | ✔ | 1 | ✔ | goal-progress | ✔ |
| KinoPaxSTARCleanCost | weighted (`h_costWeight_`), capped (`h_acceptCapMul_`) | 0 by default (`h_r2SeedAccept_`; true free pass when on) | ✔ | — | ✔ |
| KinoPaxSTARCOMBO | 3 sigmoids (`h_kCoverage_`/`h_kCollision_`/`h_kCost_`), budget-controlled (no cap) | 0 (removed) | ✔ | — | ✔ |
| KinoPaxSTARTrue | capped (`h_syclopCap_`) | 1 | ✔ | cost-guarded | ✔ |
| KinoPaxSTARTrueWeightedCost | weighted (`h_costWeight_`) | 1 | ✔ | cost-guarded | ✔ |
| KinoPaxSTARcostprune | capped (`h_acceptCap_`) | 1 | ✔ | cost | ✔ |
| KinoPaxSTARWeightedCost | weighted (`h_costWeight_`) | 1 (gated) | ✔ | — | ✔ |
| KinoPaxSTARnoseed | ✔ | 0 | ✔ | — | ✔ |
| KinoPaxSTARsparsefill | ✔ | ramp 0→1 | ✔ | — | ✔ |
| KinoPaxSTARcostprunenoseed | capped (`h_acceptCap_`) | 0 (`h_pSeed_`) | ✔ | cost | ✔ |
| CountingStars | ONE GLOBAL BUDGET filled in priority order: optimal (uncapped) → freshest by region ordinality → region-best guarantee → uniform draw | n/a (the R2 door is gone; ordinality is the novelty signal) | ✔ | — | ✔ |
