#!/usr/bin/env python3
"""Assert the CountingStars sweep's three files agree on every series label.

    examples/gpu/countingstars_sweep.cu        writes the CSVs
    scripts/run_countingstars_sweep.sh         chooses the deltas and per-delta flags
    scripts/process_countingstars_and_plot.m   decides which CSVs to load

WHY THIS EXISTS. When these drift, MATLAB does not error -- loadRuns() silently finds no files and
reports "0 runs" for the orphaned series, so the plot comes out looking merely sparse. That failure
mode has cost whole sweeps. Everything below is PARSED from the three real files; nothing is
restated by hand, because a hand-restated grid is just a fourth thing to drift.

This sweep has a second drift axis: the two finer deltas COULD run --only-kinopaxplus (the .sh
supports it via DELTA_EXTRA_ARGS), so at those deltas the CountingStars / KPAX / KinoPaxSTARTrue
series would not exist at all. The .sh says so with DELTA_EXTRA_ARGS, the .m says so with
deltaPlusOnly, and they have to agree or the plot expects series the sweep never wrote. (This pass
runs all deltas with an empty flag string, so plus_only is False everywhere today -- but the
mechanism is still checked.)

v3.5: BACK TO A GRID, ON ONE AXIS. CountingStars' hopeless guard (h_hopelessGuard_) was swept
on/off at a single fixed (bufferSlope, bufferFloor, explore_frac, cost_frac) point and confirmed to
help; countingstars_sweep.cu now runs it PERMANENTLY ON (CS_HOPELESS_GUARD, a scalar, not an axis
any more) and re-sweeps bufferSlope x bufferFloor instead (CS_BUFFER_SLOPES / CS_BUFFER_FLOORS,
arrays again), since a ramp tuned against the old, unguarded rule is not guaranteed to still be
best. explore_frac/cost_frac (CS_EXPLORE_FRAC / CS_COST_FRAC) stay fixed scalars. This script's job
is now: (1) confirm the .cu's bufferSlope/bufferFloor grid and hopelessGuard scalar match the .m's
copies of them, (2) confirm the two KinoPaxSTARTrue points (ANCESTOR_PRUNE_VALUES/SYCLOP_CAP in the
.cu, trueAncestorPruneValues/trueCap in the .m) agree, and (3) run the same end-to-end filename
check as before (assertion 4), which is label-agnostic and needs no changes to cover the new series.

Run from anywhere:  python scripts/cross_check_countingstars_grid.py
Exit 0 = GRIDS MATCH, 1 = GRIDS DIVERGE.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CU = os.path.join(ROOT, 'examples', 'gpu', 'countingstars_sweep.cu')
SH = os.path.join(ROOT, 'scripts', 'run_countingstars_sweep.sh')
M = os.path.join(ROOT, 'scripts', 'process_countingstars_and_plot.m')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


cu, sh, m = read(CU), read(SH), read(M)


# ---------------------------------------------------------------- parsers
def cu_array(name, ctype='float'):
    mo = re.search(r'static const %s\s+%s\[\]\s*=\s*\{([^}]*)\}' % (ctype, name), cu)
    if not mo:
        sys.exit('FATAL: %s[] not found in %s' % (name, CU))
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', mo.group(1))]


def cu_scalar(name, ctype='float'):
    mo = re.search(r'static const %s\s+%s\s*=\s*(-?\d+\.?\d*)f?' % (ctype, name), cu)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CU))
    return float(mo.group(1))


def sh_array(name):
    """Anchored at ^ so the commented-out alternate blocks in the .sh are correctly ignored."""
    mo = re.search(r'^%s=\(([^)]*)\)' % name, sh, re.M)
    if not mo:
        sys.exit('FATAL: %s=(...) not found in %s' % (name, SH))
    return re.findall(r'"([^"]*)"', mo.group(1))


def m_ints(name):
    mo = re.search(r'^%s\s*=\s*\[([^\]]*)\]' % name, m, re.M)
    if not mo:
        sys.exit('FATAL: %s = [...] not found in %s' % (name, M))
    return [int(round(float(x))) for x in re.findall(r'-?\d+\.?\d*', mo.group(1))]


def m_scalar_int(name):
    mo = re.search(r'^%s\s*=\s*(-?\d+\.?\d*)' % name, m, re.M)
    if not mo:
        sys.exit('FATAL: %s = ... not found in %s' % (name, M))
    return int(round(float(mo.group(1))))


def m_cellstr(name):
    mo = re.search(r"^%s\s*=\s*\{([^}]*)\}" % name, m, re.M)
    if not mo:
        sys.exit('FATAL: %s = {...} not found in %s' % (name, M))
    return re.findall(r"'([^']*)'", mo.group(1))


def m_bools(name):
    mo = re.search(r'^%s\s*=\s*\[([^\]]*)\]' % name, m, re.M)
    if not mo:
        sys.exit('FATAL: %s = [...] not found in %s' % (name, M))
    return [w == 'true' for w in re.findall(r'true|false', mo.group(1))]


def sh_config_int(name):
    """Read a #define out of the write_config heredoc in the .sh.

    MAX_TREE_SIZE is written into config.h by this script rather than living in the repo's checked-in
    config -- so the only honest place to read it for the derived-B assertion is the heredoc that
    writes it. (B's OTHER input, the ramp's fill_iters denominator, is CS_RAMP_FILL_ITERS -- a
    benchmark constant in countingstars_sweep.cu, not a config.h #define; see cu_scalar's use of it
    below, not this helper.)
    """
    mo = re.search(r'^#define\s+%s\s+(\d+)' % re.escape(name), sh, re.M)
    if not mo:
        sys.exit('FATAL: #define %s not found in %s write_config' % (name, SH))
    return int(mo.group(1))


def tok(x):
    """The label token convention for most fractional axes: round(100 x float)."""
    return int(round(100.0 * x))


def ftok(x):
    """The SHARE axes' token: round(1000 x float), letters `ef` and `cf`.

    1000x because a grid once reached 0.001, which rounds to the token 0 at 100x -- unreadable, and
    indistinguishable from a genuine share of 0. bufferSlope/bufferFloor use tok() (100x) instead,
    matching v3's fill_frac convention: both are coarse axes and `bs120`/`bf30` read directly as
    1.2/0.3 where `bs1200`/`bf300` would not.
    """
    return int(round(1000.0 * x))


# ---------------------------------------------------------------- the C++ side
# CountingStars: bufferSlope x bufferFloor GRID -- see the module docstring.
cu_slopes = cu_array('CS_BUFFER_SLOPES')
cu_floors = cu_array('CS_BUFFER_FLOORS')
cu_efrac = cu_scalar('CS_EXPLORE_FRAC')
cu_cfrac = cu_scalar('CS_COST_FRAC')

# KinoPaxSTARTrue: two fixed points (syclopCap pinned, ancestorPrune varies). Both constants live
# inside runKinoPaxSTARTrueBenchmark() in the .cu, but cu_array/cu_scalar search the whole file's
# text, not a scoped function body, so this finds them regardless of nesting.
cu_true_cap = cu_scalar('SYCLOP_CAP')
cu_true_anc = [int(v) for v in cu_array('ANCESTOR_PRUNE_VALUES', ctype='int')]

# v3.5: THE HOPELESS GUARD -- PERMANENTLY ON (a scalar, not a swept axis any more) at every
# CountingStars point above.
cu_hopeless = int(cu_scalar('CS_HOPELESS_GUARD', ctype='int'))

sh_deltas = sh_array('DELTA_LABELS')
sh_extra = sh_array('DELTA_EXTRA_ARGS')
# The quoted-string regex drops empty entries, so pad from the FRONT: index 0 is the full sweep and
# is the one with an empty flag string.
while len(sh_extra) < len(sh_deltas):
    sh_extra.insert(0, '')
sh_plus_only = ['--only-kinopaxplus' in e for e in sh_extra]
sh_metrics = sh_array('COST_LABELS')

problems = []

# --- Assertion 1: the grid's axes must stay in their meaningful ranges.
#
# B_frac = slope*x + floor must stay non-negative (a negative slope or floor would let B go
# negative, which the code's floor-at-1 clamp would silently turn into a positive B that looks
# fine); explore_frac and cost_frac are SHARES OF B, so each must be in [0, 1] on its own.
if any(v < 0.0 for v in cu_slopes) or any(v < 0.0 for v in cu_floors):
    problems.append('CS_BUFFER_SLOPES=%s / CS_BUFFER_FLOORS=%s has a negative entry -- B_frac = '
                    'slope*x + floor could go negative, and the planner\'s floor-at-1 clamp would '
                    'silently mask it' % (cu_slopes, cu_floors))
if not (0.0 <= cu_efrac <= 1.0) or not (0.0 <= cu_cfrac <= 1.0):
    problems.append('CS_EXPLORE_FRAC=%g / CS_COST_FRAC=%g has an entry outside [0, 1] -- each is a '
                    'share of B, not a count' % (cu_efrac, cu_cfrac))

# --- Assertion 2: react_frac = 1 - explore_frac - cost_frac MUST STAY NON-NEGATIVE. The planner
# floors it at 0, so an oversubscribed pair does not crash -- it silently switches the uniform DRAW
# off.
if cu_efrac + cu_cfrac > 1.0 + 1e-6:
    problems.append('OVERSUBSCRIBED BUDGET: explore_frac + cost_frac = %g > 1, so react_frac would '
                    'be negative and the draw silently switches off' % (cu_efrac + cu_cfrac))

# --- Assertion 2b: informational only, not a "problems" check -- logs the ramp's minimum (at
# x = 0, i.e. bufferFloor alone) for each bufferFloor so a reader can see it without re-deriving it.
#
# THE DENOMINATOR IS CS_RAMP_FILL_ITERS, NOT MAX_ITER. benchmarkCountingStars() sets
# planner.h_fillIters_ = CS_RAMP_FILL_ITERS explicitly before every run (it no longer relies on
# CountingStars' MAX_ITER-defaulted field) -- see that function's own comment for why: at
# MAX_TREE_SIZE=3,000,000 and a 10s timeout, a real run only completes ~700 iterations, well short
# of MAX_ITER, so leaving h_fillIters_ at the class default would make this preview describe a ramp
# the benchmark never actually runs.
cfg_tree = sh_config_int('MAX_TREE_SIZE')
cfg_fill_iters = int(cu_scalar('CS_RAMP_FILL_ITERS', ctype='int'))
ramp_min_info = ', '.join('floor(%g * %d / %d) = %d' % (fl, cfg_tree, cfg_fill_iters, int(fl * cfg_tree / cfg_fill_iters))
                          for fl in cu_floors)


def cs_label(slope, floor, efrac, cfrac, hg):
    """Mirrors countingStarsLabel() in the benchmark."""
    return 'CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d' % (tok(slope), tok(floor), ftok(efrac), ftok(cfrac), hg)


def true_label(cap, anc):
    """Mirrors trueLabel() in the benchmark."""
    return 'KinoPaxSTARTrue_cap%d_anc%d' % (tok(cap), anc)


cu_pairs = set()
for d, plus_only in zip(sh_deltas, sh_plus_only):
    if not plus_only:
        for slope in cu_slopes:
            for floor in cu_floors:
                cu_pairs.add((cs_label(slope, floor, cu_efrac, cu_cfrac, cu_hopeless), d))
        cu_pairs.add(('KPAX', d))
        for anc in cu_true_anc:
            cu_pairs.add((true_label(cu_true_cap, anc), d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_slopes = m_ints('csBufferSlopes')
m_floors = m_ints('csBufferFloors')
m_efrac = m_scalar_int('csExploreFrac')
m_cfrac = m_scalar_int('csCostFrac')
m_hopeless = m_scalar_int('csHopelessGuard')
m_true_cap = m_scalar_int('trueCap')
m_true_anc = m_ints('trueAncestorPruneValues')
m_deltas = m_cellstr('deltas')
m_plus_only = m_bools('deltaPlusOnly')

m_pairs = set()
for d, plus_only in zip(m_deltas, m_plus_only):
    if not plus_only:
        for slope in m_slopes:
            for floor in m_floors:
                m_pairs.add(('CountingStars_bs%d_bf%d_ef%d_cf%d_hg%d' % (slope, floor, m_efrac, m_cfrac, m_hopeless), d))
        m_pairs.add(('KPAX', d))
        for anc in m_true_anc:
            m_pairs.add(('KinoPaxSTARTrue_cap%d_anc%d' % (m_true_cap, anc), d))
    m_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- diff
only_cu = sorted(cu_pairs - m_pairs)
only_m = sorted(m_pairs - cu_pairs)

# --- Assertion 3: the grid axes themselves must agree between .cu and .m, not just their
# resulting label sets (which the diff above already checks) -- this pins down WHICH axis drifted
# when it does.
if sorted(tok(v) for v in cu_slopes) != sorted(m_slopes):
    problems.append('BUFFERSLOPE GRID DRIFT: .cu CS_BUFFER_SLOPES -> %s != .m csBufferSlopes %s'
                    % (sorted(tok(v) for v in cu_slopes), sorted(m_slopes)))
if sorted(tok(v) for v in cu_floors) != sorted(m_floors):
    problems.append('BUFFERFLOOR GRID DRIFT: .cu CS_BUFFER_FLOORS -> %s != .m csBufferFloors %s'
                    % (sorted(tok(v) for v in cu_floors), sorted(m_floors)))
if (ftok(cu_efrac), ftok(cu_cfrac)) != (m_efrac, m_cfrac):
    problems.append('EXPLORE/COST FRAC DRIFT: .cu (ef%d, cf%d) != .m (ef%d, cf%d)'
                    % (ftok(cu_efrac), ftok(cu_cfrac), m_efrac, m_cfrac))
if cu_hopeless != m_hopeless:
    problems.append('HOPELESS GUARD DRIFT: .cu CS_HOPELESS_GUARD=%d != .m csHopelessGuard=%d'
                    % (cu_hopeless, m_hopeless))
if tok(cu_true_cap) != m_true_cap:
    problems.append('KINOPAXSTARTRUE CAP DRIFT: .cu SYCLOP_CAP -> cap%d != .m trueCap=%d'
                    % (tok(cu_true_cap), m_true_cap))
if sorted(cu_true_anc) != sorted(m_true_anc):
    problems.append('KINOPAXSTARTRUE ANCESTOR_PRUNE DRIFT: .cu ANCESTOR_PRUNE_VALUES %s != '
                    '.m trueAncestorPruneValues %s' % (cu_true_anc, m_true_anc))

if sh_deltas != m_deltas:
    problems.append('DELTA_LABELS %s (%s) != deltas %s (%s)' % (sh_deltas, SH, m_deltas, M))
if sh_plus_only != m_plus_only:
    problems.append('--only-kinopaxplus flags %s (%s) != deltaPlusOnly %s (%s)'
                    % (sh_plus_only, SH, m_plus_only, M))

# --- Assertion 4: THE FILENAMES MUST MATCH END TO END.
#
# Matching label SETS is not enough, and this is the assertion that would have caught the two bugs
# that actually shipped. Both sides agreed perfectly on the label `CountingStars_r0_h1_e300` while:
#
#   * writePerIterationCSV()'s planner-name whitelist did not know it, so it fell through to the
#     KinoPaxPlus arm -- which keys on the DELTA and omits build_delta entirely. The length and
#     effort builds then wrote the SAME path, and the second overwrote the first.
#   * loadRuns()'s whitelist did not know it either, so the plot script error()d on it.
#
# So model both filename constructions from their real source, and diff the resulting PATHS. This
# assertion is label-agnostic (it iterates m_pairs and re-derives the prefix whitelist from the
# actual function bodies), so it needs no changes to cover the new KinoPaxSTARTrue series.

# Parse the benchmark's writer whitelist rather than restating it.
wmo = re.search(r'void writePerIterationCSV\(.*?\n\}', cu, re.S)
if not wmo:
    problems.append('writePerIterationCSV() not found in %s' % CU)
    cu_writer_prefixes = []
else:
    cu_writer_prefixes = re.findall(r'delta_label\.rfind\("([^"]+)",\s*0\)\s*==\s*0', wmo.group(0))
    if not cu_writer_prefixes:
        problems.append('writePerIterationCSV() exposes no rfind() prefixes -- parser needs updating')

# Parse the plot script's loader whitelist the same way.
lmo = re.search(r'function runs = loadRuns\(.*?\nend\n', m, re.S)
if not lmo:
    problems.append('loadRuns() not found in %s' % M)
    m_loader_exact, m_loader_prefixes = set(), []
else:
    loader = lmo.group(0)
    m_loader_exact = set(re.findall(r"case\s+'([^']+)'", loader))
    m_loader_prefixes = re.findall(r"startsWith\(planner,\s*'([^']+)'\)", loader)
    if not m_loader_prefixes and not m_loader_exact:
        problems.append('loadRuns() exposes no case/startsWith arms -- parser needs updating')


def cu_filename(label, delta_metric, run):
    """Mirrors writePerIterationCSV() in the benchmark.

    The dispatch is on RunResult::delta_label, which is not always the series name. KinoPaxPlus sets
    delta_label to the BUILD TOKEN (benchmarkKinoPaxPlus: `result.delta_label = deltaLabel`), which
    is what routes it to the delta-keyed arm and keeps its two discretisations in separate files.
    Every other arm sets delta_label to its series label.
    """
    if label == 'KPAX':
        return 'ENV_KPAX_delta%s_run%d.csv' % (delta_metric, run)
    effective = delta_metric if label == 'KinoPaxPlus' else label
    if cu_writer_prefixes and effective.startswith(tuple(cu_writer_prefixes)):
        return 'ENV_%s_delta%s_run%d.csv' % (effective, delta_metric, run)
    return 'ENV_delta%s_run%d.csv' % (effective, run)


def m_filename(label, delta_metric, run):
    """Mirrors loadRuns() in the plot script. None where it would error()."""
    if label == 'KinoPaxPlus':
        return 'ENV_delta%s_run%d.csv' % (delta_metric, run)
    if label == 'KPAX':
        return 'ENV_KPAX_delta%s_run%d.csv' % (delta_metric, run)
    if m_loader_prefixes and label.startswith(tuple(m_loader_prefixes)):
        return 'ENV_%s_delta%s_run%d.csv' % (label, delta_metric, run)
    return None


if cu_writer_prefixes and (m_loader_prefixes or m_loader_exact):
    for lbl, d in sorted(m_pairs):
        dm = '%s_%s' % (d, sh_metrics[0])
        written = cu_filename(lbl, dm, 0)
        wanted = m_filename(lbl, dm, 0)
        if wanted is None:
            problems.append('loadRuns() would ERROR on "%s" -- no case arm and no matching prefix '
                            'in %s' % (lbl, sorted(set(m_loader_prefixes))))
        elif written != wanted:
            problems.append('FILENAME MISMATCH for "%s" [%s]: benchmark writes %s but plot script '
                            'wants %s' % (lbl, dm, written, wanted))

    # A label routed to the delta-keyed arm omits build_delta, so every cost metric collides on one
    # path and the second build silently overwrites the first. KinoPaxPlus is exempt: it keys on the
    # delta BY DESIGN, and the delta token already carries the metric.
    for lbl, d in sorted(m_pairs):
        if lbl == 'KinoPaxPlus':
            continue
        names = {cu_filename(lbl, '%s_%s' % (d, mt), 0) for mt in sh_metrics}
        if len(names) < len(sh_metrics):
            problems.append('COST-METRIC COLLISION for "%s" [%s]: every metric writes %s, so the '
                            'second build overwrites the first' % (lbl, d, names.pop()))

print('cost metrics : %s' % ', '.join(sh_metrics))
print('deltas       : %s  (--only-kinopaxplus: %s)'
      % (', '.join(sh_deltas), ', '.join(str(b) for b in sh_plus_only)))
print('CountingStars grid        : bufferSlope %s x bufferFloor %s, explore_frac=%g cost_frac=%g, '
      'hopelessGuard=%d (permanent)' % (cu_slopes, cu_floors, cu_efrac, cu_cfrac, cu_hopeless))
print('KinoPaxSTARTrue points    : syclopCap=%g x ancestorPrune %s' % (cu_true_cap, cu_true_anc))
print('series (.cu) : %d' % len(cu_pairs))
print('series (.m)  : %d' % len(m_pairs))
print('ramp minimum : B(x=0) at bufferFloor -- %s' % ramp_min_info)

if only_cu:
    print('\nWritten by the benchmark but NEVER LOADED by the plot script (%d):' % len(only_cu))
    for lbl, d in only_cu:
        print('  %-46s [%s]' % (lbl, d))
if only_m:
    print('\nExpected by the plot script but NEVER WRITTEN by the benchmark (%d):' % len(only_m))
    for lbl, d in only_m:
        print('  %-46s [%s]' % (lbl, d))
if problems:
    print('\nOther problems (%d):' % len(problems))
    for p in problems:
        print('  ' + p)

ok = not only_cu and not only_m and not problems
print('\n%s' % ('GRIDS MATCH' if ok else 'GRIDS DIVERGE'))
sys.exit(0 if ok else 1)
