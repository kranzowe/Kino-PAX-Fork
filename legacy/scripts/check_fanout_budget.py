#!/usr/bin/env python3
"""Assert COMBO's fan-out block arithmetic can never force the slow kernel2 propagate path.

WHY THIS EXISTS. propagateFrontier drops onto kernel2 when

    frontierRepeatSize * activeBlockSize > MAX_TREE_SIZE - treeSize

and the previous fan-out rule hit that sporadically in early iterations. The reason was not tuning:
repHi was solved from a SURPLUS against an ESTIMATED frontier and an estimated favoured count, then
clamped at h_repeatMax_ -- and the moment that clamp bound, the identity the whole scheme rested on
(sum(rep) = budgetBlocks) stopped holding, so sum(rep) was no longer related to the ceiling it had
been fitted under. Whether kernel2 fired came down to how far the two estimates happened to be off.

The rule this replaces it with solves for repHi directly, against a COUNTED frontier and a COUNTED
favoured set, so the block total is known before the launch. This file replicates that host
arithmetic exactly -- see the block after findInd in src/planners/KinoPaxSTARCOMBO.cu -- and sweeps
it, because there is no nvcc on the authoring machine and this is the only way to check the property
that actually matters without hardware.

THE INVARIANT:

    sum(rep) = F + (repHi - 1) * nFav  <=  max(F, blockCeiling)

The `max` is the honest part. Whenever blockCeiling >= F the fan-out fits inside the budget with
room to spare. When blockCeiling < F the budget is already spent by the rep >= 1 floor alone, repHi
collapses to 1, and sum(rep) = F -- so kernel2 becomes unavoidable exactly when
activeBlockSize * F > remaining, which is a property of the FRONTIER (F >= nActive, because Part B
reactivates every region's best unconditionally) and not of any fan-out rule. That bound is
documented in the planner and is the only route to kernel2 left.

Run from anywhere:  python scripts/check_fanout_budget.py
Exit 0 = BUDGET SOUND, 1 = BUDGET UNSOUND.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CU = os.path.join(ROOT, 'src', 'planners', 'KinoPaxSTARCOMBO.cu')
CFG = os.path.join(ROOT, 'include', 'config', 'config.h')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


cu, cfg = read(CU), read(CFG)


def cu_member(name):
    """Pull a constructor default straight out of the planner, so this never drifts from it."""
    mo = re.search(r'^\s*%s\s*=\s*([-\d.]+)f?\s*;' % re.escape(name), cu, re.M)
    if not mo:
        sys.exit('FATAL: %s default not found in %s' % (name, CU))
    return float(mo.group(1))


def cfg_int(name):
    mo = re.search(r'^#define\s+%s\s+(\d+)' % re.escape(name), cfg, re.M)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CFG))
    return int(mo.group(1))


MAX_TREE_SIZE = cfg_int('MAX_TREE_SIZE')
MAX_ITER = cfg_int('MAX_ITER')
SELECTIVITY = cu_member('h_selectivity_')
REPEAT_MAX = cu_member('h_repeatMax_')
GROWTH_EXP = cu_member('h_growthExp_')
BLOCK_SIZE = int(cu_member('h_activeBlockSize_'))
GROWTH_ITERS = MAX_ITER   # h_growthIters_ = MAX_ITER in the constructor

# The literal 0.8 margin against the kernel1 condition, read from the planner rather than restated.
mo = re.search(r'fminf\(h_selectivity_ \* h_wantThisIter_ / nu,\s*([\d.]+)f \* remaining\)', cu)
if not mo:
    sys.exit('FATAL: the blockCeiling expression in %s no longer matches this script' % CU)
MARGIN = float(mo.group(1))


def combo_want_this_iter(remaining, itr, growth_iters, growth_exp):
    """Mirrors comboWantThisIter() in the planner."""
    if growth_iters <= 0.0:
        return remaining
    u0 = min(1.0, max(0.0, itr / growth_iters))
    u1 = min(1.0, max(0.0, (itr + 1.0) / growth_iters))
    inv = (1.0 / growth_exp) if growth_exp > 0.0 else 1.0
    s0 = u0 ** inv
    s1 = u1 ** inv
    headroom = 1.0 - s0
    if headroom <= 1e-6:
        return remaining
    return remaining * min(1.0, (s1 - s0) / headroom)


def solve(tree_size, itr, frontier_size, n_fav, collision_frac):
    """Mirrors the block after findInd in KinoPaxSTARCOMBO::propagateFrontier."""
    remaining = max(0.0, float(MAX_TREE_SIZE) - float(tree_size))
    nu = (1.0 - collision_frac) if 0.0 < collision_frac < 1.0 else 0.9
    want = combo_want_this_iter(remaining, float(itr), float(GROWTH_ITERS), GROWTH_EXP)
    ceiling = min(SELECTIVITY * want / nu, MARGIN * remaining) / float(BLOCK_SIZE)

    b_max = 1.0 + (ceiling - float(frontier_size)) / float(n_fav) if n_fav > 0 else 1.0
    rep_hi = int(max(1.0, min(REPEAT_MAX, math.floor(b_max))))
    sum_rep = frontier_size + (rep_hi - 1) * n_fav
    return ceiling, rep_hi, sum_rep


# ---------------------------------------------------------------- sweep
problems = []
cases = 0

TREE_FRACS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
ITERS = (1, 2, 3, 5, 10, 30, 100, 200, 299, 300)
FRONTIERS = (1, 2, 10, 100, 1000, 5000, 12500, 40000, 100000, 399999)
FAV_FRACS = (0.0, 0.001, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)   # 1.0 = the degenerate/uniform branch
COLLISION = (0.0, 0.1, 0.5, 0.9, 1.0)

for tf in TREE_FRACS:
    tree_size = int(tf * MAX_TREE_SIZE)
    for itr in ITERS:
        for f in FRONTIERS:
            if f > MAX_TREE_SIZE:
                continue
            for ff in FAV_FRACS:
                n_fav = int(round(ff * f))
                for cfrac in COLLISION:
                    cases += 1
                    ceiling, rep_hi, sum_rep = solve(tree_size, itr, f, n_fav, cfrac)
                    ctx = ('tree=%d itr=%d F=%d nFav=%d cf=%g -> ceiling=%.1f repHi=%d sumRep=%d'
                           % (tree_size, itr, f, n_fav, cfrac, ceiling, rep_hi, sum_rep))

                    # --- repHi stays inside its declared range ---
                    if rep_hi < 1:
                        problems.append('repHi < 1: ' + ctx)
                    if rep_hi > REPEAT_MAX:
                        problems.append('repHi > h_repeatMax_: ' + ctx)

                    # --- THE INVARIANT: the only thing that may exceed the ceiling is the floor ---
                    if sum_rep > max(float(f), ceiling) + 1e-6:
                        problems.append('sum(rep) exceeds max(F, ceiling): ' + ctx)

                    # --- kernel1 is retained whenever the frontier alone leaves room for it ---
                    remaining = max(0.0, float(MAX_TREE_SIZE) - float(tree_size))
                    if f * BLOCK_SIZE <= remaining and sum_rep * BLOCK_SIZE > remaining:
                        problems.append('KERNEL2 FORCED although the frontier fits: ' + ctx)

                    # --- the boost only ever costs blocks the ceiling had spare ---
                    if rep_hi > 1 and ceiling < f:
                        problems.append('boosted with no headroom (ceiling < F): ' + ctx)

# --- the degenerate branch is the uniform-rep control arm: nFav == F must spend the ceiling, not
# some fraction of it. This is what makes kFan = 0 a real arm rather than a special case. ---
for tf in TREE_FRACS:
    tree_size = int(tf * MAX_TREE_SIZE)
    for f in (1, 100, 10000, 100000):
        ceiling, rep_hi, sum_rep = solve(tree_size, 5, f, f, 0.1)
        cases += 1
        if ceiling >= f and rep_hi < min(REPEAT_MAX, math.floor(ceiling / f)):
            problems.append('degenerate branch underspends: tree=%d F=%d ceiling=%.1f repHi=%d'
                            % (tree_size, f, ceiling, rep_hi))

print('MAX_TREE_SIZE %d   MAX_ITER %d   blockSize %d   margin %g'
      % (MAX_TREE_SIZE, MAX_ITER, BLOCK_SIZE, MARGIN))
print('selectivity %g   repeatMax %g   growthExp %g' % (SELECTIVITY, REPEAT_MAX, GROWTH_EXP))
print('cases checked : %d' % cases)

if problems:
    shown = problems[:20]
    print('\nProblems (%d, first %d shown):' % (len(problems), len(shown)))
    for p in shown:
        print('  ' + p)

print('\n%s' % ('BUDGET SOUND' if not problems else 'BUDGET UNSOUND'))
sys.exit(0 if not problems else 1)
