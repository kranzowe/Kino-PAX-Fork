#!/usr/bin/env python3
"""Assert CountingStars' three histogram selections are exact top-X selections.

WHY THIS EXISTS. v3 added a second selection door -- CHEAPEST, over the cost distance -- and v3.1 a
third, the reactivation arm over dormant tree nodes. All three are built the way the freshness door
already was: bucket the value, histogram it, exclusive-scan the histogram to a cutoff, and spend the
fractional remainder with a boundary roll. Two properties have to hold for that to be a top-X
selection rather than an approximation, and neither is visible in a diff:

  1. THE BUCKET MAP MUST BE MONOTONE. The freshness door's is a clamp and obviously is. The cost
     door's is LOG, anchored at distMax, and picked over a linear map because a distance is
     (cost - regionMin)/costScale and therefore piles up near 0 with a long tail. Monotone is the
     only property the scan needs from it, and log is easy to get subtly wrong at the ends.
  2. THE SOLVE MUST ADMIT EXACTLY min(X, n) IN EXPECTATION. Not approximately: the boundary roll
     exists precisely because taking the whole boundary bucket overshoots by up to one bucket's
     width, and where thousands of candidates share a bucket that is most of a frontier.

Neither can be checked on the authoring machine any other way -- there is no nvcc here, and a run
would only show a door that quietly under- or over-spends its share of the budget as "the tuning is
off". So this replicates the real arithmetic and sweeps it.

It also checks two things around the histograms that have the same "fails silently" character:

  3. THE SLOT LAYOUT. v3.1 put a THIRD histogram -- the reactivation one, over dormant tree nodes --
     into the same buffer, which now carries three 256-bucket ranges and two scalar slots. An
     off-by-one in a base would have one histogram's votes land in another's buckets, and nothing
     downstream would notice: every cutoff would still solve, just against the wrong counts.
  4. THE COMPLETENESS FLOOR IS NON-ZERO. Part B's whole reactivation budget goes to the cheapest
     dormant nodes, and on its own that is not probabilistically complete -- a node's distance has a
     fixed numerator over a non-increasing region minimum, so it only ever grows and a node once
     above the cutoff can NEVER come back. h_reactFloor_ is what restores completeness, and it looks
     exactly like a magic constant somebody would tidy away. This is the check that stops them.

And the budget the solves are handed: react_frac = 1 - explore_frac - cost_frac is floored at 0 in
the planner, so an oversubscribed grid point does not fail -- it silently switches reactivation off.
cross_check_countingstars_grid.py asserts the same thing from the grid side; this one asserts it
against the arithmetic.

Everything numeric is PARSED from the real sources. A hand-restated constant is just a fourth place
to drift.

Run from anywhere:  python scripts/check_countingstars_histogram.py
Exit 0 = SELECTION SOUND, 1 = SELECTION UNSOUND.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUH = os.path.join(ROOT, 'include', 'planners', 'CountingStars.cuh')
CU = os.path.join(ROOT, 'src', 'planners', 'CountingStars.cu')
SWEEP = os.path.join(ROOT, 'examples', 'gpu', 'countingstars_sweep.cu')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


cuh, cu, sweep = read(CUH), read(CU), read(SWEEP)


def cuh_const(name, ctype='int'):
    mo = re.search(r'static const %s\s+%s\s*=\s*(-?\d+\.?\d*)f?\s*;' % (ctype, name), cuh)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CUH))
    return float(mo.group(1))


def cu_array(text, name, path):
    mo = re.search(r'static const float %s\[\]\s*=\s*\{([^}]*)\}' % name, text)
    if not mo:
        sys.exit('FATAL: %s[] not found in %s' % (name, path))
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', mo.group(1))]


ORD_BUCKETS = int(cuh_const('CS_ORD_BUCKETS'))
COST_BUCKETS = int(cuh_const('CS_COST_BUCKETS'))
LOG_SCALE = cuh_const('CS_COST_LOG_SCALE', 'float')


def cuh_slot(name):
    """Slot bases are defined in terms of the bucket counts, so evaluate the real expression rather
    than matching a literal -- that is the whole point of checking them."""
    mo = re.search(r'static const int\s+%s\s*=\s*([^;]+);' % name, cuh)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CUH))
    expr = mo.group(1).strip()
    try:
        return int(eval(expr, {'__builtins__': {}},
                        {'CS_ORD_BUCKETS': ORD_BUCKETS, 'CS_COST_BUCKETS': COST_BUCKETS}))
    except Exception as exc:
        sys.exit('FATAL: cannot evaluate %s = %s (%s)' % (name, expr, exc))


HIST_ORD_BASE = cuh_slot('CS_HIST_ORD_BASE')
HIST_OPT_SLOT = cuh_slot('CS_HIST_OPT_SLOT')
HIST_COST_BASE = cuh_slot('CS_HIST_COST_BASE')
HIST_REACT_BASE = cuh_slot('CS_HIST_REACT_BASE')
HIST_DORMANT_SLOT = cuh_slot('CS_HIST_DORMANT_SLOT')
HIST_SIZE = cuh_slot('CS_HIST_SIZE')


def cu_member_float(name):
    mo = re.search(r'^\s*%s\s*=\s*([-\d.eE+]+)f?\s*;' % re.escape(name), cu, re.M)
    if not mo:
        sys.exit('FATAL: %s default not found in %s' % (name, CU))
    return float(mo.group(1))


REACT_FLOOR = cu_member_float('h_reactFloor_')

# --- Pin the Python mirrors to the real expressions. If either is edited, this fails LOUDLY here
# rather than quietly passing a check of arithmetic the planner no longer does. ---
if not re.search(r'float b = float\(CS_COST_BUCKETS - 1\) \+ CS_COST_LOG_SCALE \* log2f\(d / distMax\);', cuh):
    sys.exit('FATAL: csCostBucket in %s no longer matches this script\'s mirror' % CUH)
if not re.search(r'pBoundary = \(h > 0\.0f\) \? fminf\(1\.0f, fmaxf\(0\.0f, \(X - acc\) / h\)\) : 0\.0f;', cu):
    sys.exit('FATAL: csSolveCutoff in %s no longer matches this script\'s mirror' % CU)
if not re.search(r'h_reactFrac_ = fmaxf\(0\.0f, 1\.0f - h_exploreFrac_ - h_costFrac_\);', cu):
    sys.exit('FATAL: the react_frac expression in %s no longer matches this script' % CU)
# v3.1: the reactivation arm must still be solved by the SAME helper over the SAME bucket map, which
# is what lets the exactness proof below cover it without a second set of cases.
if not re.search(r'csSolveCutoff\(h_acceptHistogram_ \+ CS_HIST_REACT_BASE, CS_COST_BUCKETS,', cu):
    sys.exit('FATAL: the reactivation cutoff solve in %s no longer matches this script' % CU)


# ---------------------------------------------------------------- the mirrors
def ord_bucket(ord_):
    """Mirrors csOrdBucket() in the header."""
    if ord_ < 0:
        return 0
    return ord_ if ord_ < ORD_BUCKETS else ORD_BUCKETS - 1


def c_log2f(x):
    """math.log2 with C's IEEE semantics at the ends, which is what the kernel actually executes.

    log2f(+0) is -INF in C and a ValueError in Python, and the difference is load-bearing here: the
    ratio d/distMax UNDERFLOWS to +0 for any d more than ~2^-149 of distMax (a 1e-300 distance
    against a 1e30 anchor, say, or any finite d against an infinite one). In the kernel that gives
    b = -INF, which the `!(b > 0.0f)` guard turns into bucket 0 -- the right answer, since such a
    candidate is at its region's minimum for any purpose that matters. Raising here instead would
    report a bug the kernel does not have.
    """
    if x == 0.0:
        return float('-inf')
    if x != x or x < 0.0:      # NaN or negative: log2f returns NaN, and !(b > 0) then catches it
        return float('nan')
    return math.log2(x)


def cost_bucket(d, dist_max):
    """Mirrors csCostBucket() in the header."""
    if not (d > 0.0) or not (dist_max > 0.0):
        return 0
    if d >= dist_max:
        return COST_BUCKETS - 1
    b = float(COST_BUCKETS - 1) + LOG_SCALE * c_log2f(d / dist_max)
    if not (b > 0.0):          # also the NaN guard: !(NaN > 0) is true
        return 0
    bi = int(b)
    return bi if bi < COST_BUCKETS - 1 else COST_BUCKETS - 1


def solve_cutoff(hist, n_buckets, x):
    """Mirrors csSolveCutoff() in the planner."""
    cutoff, p_boundary, acc = n_buckets, 0.0, 0.0
    for k in range(n_buckets):
        h = float(hist[k])
        if acc + h >= x:
            return k, (min(1.0, max(0.0, (x - acc) / h)) if h > 0.0 else 0.0)
        acc += h
    return cutoff, p_boundary


def expected_admitted(values, bucket_of, n_buckets, x):
    """The count pass 2 admits in expectation, given pass 1's histogram and the host's solve."""
    hist = [0] * n_buckets
    buckets = [bucket_of(v) for v in values]
    for b in buckets:
        hist[b] += 1
    cutoff, p_boundary = solve_cutoff(hist, n_buckets, x)
    below = sum(1 for b in buckets if b < cutoff)
    on = sum(1 for b in buckets if b == cutoff)
    return below + p_boundary * on, cutoff, p_boundary


problems = []
cases = 0


# ================================================================= 1. monotonicity
#
# The scan admits every candidate in a bucket strictly below the cutoff and none above it, so the
# selected set is a prefix BY BUCKET. That is a top-X selection over the underlying value if and only
# if the bucket map never decreases as the value rises.
for dist_max in (1e-30, 1e-6, 1e-3, 0.017, 1.0, 3.7, 1e3, 1e9):
    prev = -1
    # 400 points spanning 30 octaves below distMax, then straddling and passing it.
    grid = [dist_max * (2.0 ** (-30.0 + 31.0 * i / 399.0)) for i in range(400)]
    grid += [dist_max * f for f in (0.999999, 1.0, 1.000001, 2.0, 1e6)]
    grid.sort()
    for d in grid:
        cases += 1
        b = cost_bucket(d, dist_max)
        if b < prev:
            problems.append('csCostBucket NOT MONOTONE at distMax=%g d=%g: %d after %d'
                            % (dist_max, d, b, prev))
        prev = b

for a in range(-5, ORD_BUCKETS + 5):
    cases += 1
    if a > -5 and ord_bucket(a) < ord_bucket(a - 1):
        problems.append('csOrdBucket NOT MONOTONE at ord=%d' % a)


# ================================================================= 2. range, including the ends
#
# Every bucket index is used to address a fixed-size histogram from inside a kernel, so an
# out-of-range value is a stray write, not a wrong answer.
EDGE = (0.0, -0.0, -1.0, -1e30, 1e-300, 1e-30, 1e30, float('inf'))
for dist_max in (0.0, -1.0, 1e-30, 1.0, 1e30, float('inf')):
    for d in EDGE + (dist_max, dist_max * 0.5, dist_max * 2.0):
        cases += 1
        try:
            b = cost_bucket(d, dist_max)
        except (ValueError, OverflowError) as exc:
            problems.append('csCostBucket RAISED at distMax=%g d=%g: %s' % (dist_max, d, exc))
            continue
        if not (0 <= b < COST_BUCKETS):
            problems.append('csCostBucket OUT OF RANGE at distMax=%g d=%g -> %d' % (dist_max, d, b))

for a in (-(1 << 30), -1, 0, 1, ORD_BUCKETS - 1, ORD_BUCKETS, 1 << 30):
    cases += 1
    b = ord_bucket(a)
    if not (0 <= b < ORD_BUCKETS):
        problems.append('csOrdBucket OUT OF RANGE at ord=%d -> %d' % (a, b))


# ================================================================= 3. selection exactness
#
# THE PROPERTY THE BOUNDARY ROLL EXISTS FOR: the solve admits exactly min(X, n) in expectation, for
# any distribution and any X. Without the roll it would admit the whole boundary bucket and overshoot
# by up to its width.
def gen(kind, n, seed):
    """Synthetic distance distributions, including the two pathological ones."""
    st = seed
    out = []
    for _ in range(n):
        st = (1103515245 * st + 12345) & 0x7FFFFFFF
        u = (st + 1) / 2147483649.0
        if kind == 'uniform':
            out.append(u)
        elif kind == 'exponential':       # the shape a real distance distribution is expected to have
            out.append(-math.log(u) * 0.05)
        elif kind == 'lognormal':
            out.append(math.exp(-6.0 + 3.0 * (u - 0.5) * 3.4641))
        elif kind == 'onebucket':         # EVERYTHING in bucket 0 -- the degenerate case the log map
            out.append(1e-12 * u)         #   is meant to avoid, where the door becomes a uniform draw
        elif kind == 'twovalues':         # maximal ties: only two distinct buckets exist
            out.append(0.5 if u < 0.5 else 1.0)
    return out


for kind in ('uniform', 'exponential', 'lognormal', 'onebucket', 'twovalues'):
    for n in (1, 2, 37, 1000, 5000):
        vals = gen(kind, n, 7919 + n)
        dist_max = max(vals)
        for x in (0.0, 0.5, 1.0, n * 0.001, n * 0.1, n * 0.5, n - 0.5, float(n), n * 1.5, n * 100.0):
            cases += 1
            got, cutoff, p_b = expected_admitted(vals, lambda v: cost_bucket(v, dist_max),
                                                 COST_BUCKETS, x)
            want = min(x, float(n))
            if abs(got - want) > 1e-6 * max(1.0, want):
                problems.append('COST SELECTION INEXACT [%s n=%d X=%g]: expected %g admitted, got %g '
                                '(cutoff %d, pBoundary %g)' % (kind, n, x, want, got, cutoff, p_b))
            if not (0.0 <= p_b <= 1.0):
                problems.append('pBoundary OUT OF [0,1] [%s n=%d X=%g]: %g' % (kind, n, x, p_b))

        # The same solve over the ordinality door's clamp map, so the shared helper is checked on
        # both of its call sites rather than only the new one.
        ords = [min(ORD_BUCKETS + 40, int(v * 900)) for v in vals]
        for x in (0.0, 1.0, n * 0.25, float(n), n * 3.0):
            cases += 1
            got, cutoff, p_b = expected_admitted(ords, ord_bucket, ORD_BUCKETS, x)
            want = min(x, float(n))
            if abs(got - want) > 1e-6 * max(1.0, want):
                problems.append('ORD SELECTION INEXACT [%s n=%d X=%g]: expected %g, got %g'
                                % (kind, n, x, want, got))


# ================================================================= 4. the ablation arms
#
# explore_frac = 0 and cost_frac = 0 are grid points, and the sweep treats them as real control arms
# rather than special cases the caller has to guard. That rests entirely on X = 0 returning a cutoff
# nothing can be below and a pBoundary of 0.
for kind in ('uniform', 'exponential', 'onebucket'):
    vals = gen(kind, 500, 104729)
    dist_max = max(vals)
    cases += 1
    got, cutoff, p_b = expected_admitted(vals, lambda v: cost_bucket(v, dist_max), COST_BUCKETS, 0.0)
    if got != 0.0 or cutoff != 0 or p_b != 0.0:
        problems.append('X = 0 DOES NOT ABLATE [%s]: admitted %g (cutoff %d, pBoundary %g) -- the '
                        'sweep\'s frac = 0 arms are not controls' % (kind, got, cutoff, p_b))

    # And the mirror end: X >= n must saturate to "admit everything", via the loop falling through to
    # cutoff = nBuckets, which no bucket index can equal.
    cases += 1
    got, cutoff, p_b = expected_admitted(vals, lambda v: cost_bucket(v, dist_max), COST_BUCKETS,
                                         len(vals) * 2.0)
    if abs(got - len(vals)) > 1e-9:
        problems.append('X >= n DOES NOT SATURATE [%s]: admitted %g of %d (cutoff %d)'
                        % (kind, got, len(vals), cutoff))


# ================================================================= 5. the budget the solves are given
#
# react_frac is floored at 0 in the planner, so an oversubscribed pair does not fail -- it silently
# switches the uniform draw off, and two grid points that differ only in how far past 1 they went
# would produce identical runs under different labels.
#
# v3.2: FILL_FRACS is gone -- B is now a per-iteration ramp over BUFFER_SLOPES/BUFFER_FLOORS, parsed
# below purely for the grid summary line. Neither ever entered the oversubscription math (only
# explore_frac + cost_frac do), so dropping the old outer loop over the fill/buffer axis changes
# nothing about what this check verifies -- explore_frac/cost_frac are now single-element arrays
# (fixed at 0.3 each), so this degenerates to one check, kept for robustness against a future
# re-expansion of either axis.
slopes = cu_array(sweep, 'BUFFER_SLOPES', SWEEP)
floors = cu_array(sweep, 'BUFFER_FLOORS', SWEEP)
efracs = cu_array(sweep, 'EXPLORE_FRACS', SWEEP)
cfracs = cu_array(sweep, 'COST_FRACS', SWEEP)

for ef in efracs:
    for cf in cfracs:
        cases += 1
        react = 1.0 - ef - cf
        if react < -1e-6:
            problems.append('OVERSUBSCRIBED BUDGET (explore %g, cost %g): react_frac '
                            '%g < 0, so the draw is silently switched off' % (ef, cf, react))
        if ef + cf > 1.0 + 1e-6:
            problems.append('SHARES SUM ABOVE 1 (explore %g + cost %g = %g)' % (ef, cf, ef + cf))


# ================================================================= 6. the shared histogram buffer
#
# Three 256-bucket ranges and two scalar slots in one buffer. Each range must start where the last
# one ends, the scalars must not collide with any bucket, and CS_HIST_SIZE must cover all of it --
# an off-by-one puts one histogram's votes in another's buckets, and every cutoff still solves.
layout = [('ord buckets', HIST_ORD_BASE, ORD_BUCKETS),
          ('optimal slot', HIST_OPT_SLOT, 1),
          ('cost buckets', HIST_COST_BASE, COST_BUCKETS),
          ('react buckets', HIST_REACT_BASE, COST_BUCKETS),
          ('dormant slot', HIST_DORMANT_SLOT, 1)]

occupied = {}
for name, base, width in layout:
    cases += 1
    if base < 0:
        problems.append('SLOT LAYOUT: %s starts at %d, below the buffer' % (name, base))
    if base + width > HIST_SIZE:
        problems.append('SLOT LAYOUT: %s spans [%d, %d) but CS_HIST_SIZE is %d -- an atomicAdd past '
                        'the end of the buffer' % (name, base, base + width, HIST_SIZE))
    for i in range(base, base + width):
        if i in occupied:
            problems.append('SLOT LAYOUT: %s overlaps %s at index %d -- one histogram\'s votes land '
                            'in the other\'s buckets' % (name, occupied[i], i))
            break
        occupied[i] = name

cases += 1
if len(occupied) != HIST_SIZE:
    problems.append('SLOT LAYOUT: the five regions cover %d of CS_HIST_SIZE = %d entries -- %d '
                    'unused, which means a base is wrong rather than merely wasteful'
                    % (len(occupied), HIST_SIZE, HIST_SIZE - len(occupied)))


# ================================================================= 7. the completeness floor
#
# Part B's whole reactivation budget goes to the cheapest dormant nodes. That alone is NOT
# probabilistically complete, and the failure is permanent rather than transient: a node's
#
#     distance = (cost - minCostsR1[r]) / costScale
#
# has a FIXED numerator (cost is written once at insertion, there is no rewiring) over a
# NON-INCREASING minCostsR1[r], so distance only ever GROWS. A node once above the cutoff can never
# return, its subtree becomes unreachable, and any solution routed through it is lost.
#
# h_reactFloor_ is what restores "expanded infinitely often in the limit", and it looks exactly like
# a magic constant somebody would tidy away.
cases += 1
if not (REACT_FLOOR > 0.0):
    problems.append('COMPLETENESS FLOOR IS ZERO: h_reactFloor_ = %g. Part B\'s cost arm is a top-K '
                    'over a distance that only ever grows, so without a non-zero floor a node above '
                    'the cutoff is dead permanently and the planner is not probabilistically '
                    'complete.' % REACT_FLOOR)
if REACT_FLOOR >= 1.0:
    problems.append('COMPLETENESS FLOOR IS %g: at >= 1 every dormant node is reactivated every '
                    'iteration and the frontier is the whole tree.' % REACT_FLOOR)

print('histogram    : ord[%d,%d) opt[%d] cost[%d,%d) react[%d,%d) dormant[%d]  size %d'
      % (HIST_ORD_BASE, HIST_ORD_BASE + ORD_BUCKETS, HIST_OPT_SLOT,
         HIST_COST_BASE, HIST_COST_BASE + COST_BUCKETS,
         HIST_REACT_BASE, HIST_REACT_BASE + COST_BUCKETS, HIST_DORMANT_SLOT, HIST_SIZE))
print('reactFloor   : %g  (~%.0f nodes/iter at a 3e6-node tree)' % (REACT_FLOOR, REACT_FLOOR * 3e6))
print('ordBuckets %d   costBuckets %d   logScale %g  (window %.1f octaves below distMax)'
      % (ORD_BUCKETS, COST_BUCKETS, LOG_SCALE, (COST_BUCKETS - 1) / LOG_SCALE))
print('grid       : slope %s x floor %s x explore %s x cost %s' % (slopes, floors, efracs, cfracs))
print('cases checked : %d' % cases)

if problems:
    shown = problems[:20]
    print('\nProblems (%d, first %d shown):' % (len(problems), len(shown)))
    for p in shown:
        print('  ' + p)

print('\n%s' % ('SELECTION SOUND' if not problems else 'SELECTION UNSOUND'))
sys.exit(0 if not problems else 1)
