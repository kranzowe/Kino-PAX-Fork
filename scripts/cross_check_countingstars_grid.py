#!/usr/bin/env python3
"""Assert the CountingStars sweep's three files agree on every series label.

    examples/gpu/countingstars_sweep.cu        writes the CSVs
    scripts/run_countingstars_sweep.sh         chooses the deltas and per-delta flags
    scripts/process_countingstars_and_plot.m   decides which CSVs to load

WHY THIS EXISTS. When these drift, MATLAB does not error -- loadRuns() silently finds no files and
reports "0 runs" for the orphaned series, so the plot comes out looking merely sparse. That failure
mode has cost whole sweeps. Everything below is PARSED from the three real files; nothing is
restated by hand, because a hand-restated grid is just a fourth thing to drift.

This sweep has a second drift axis the COMBO one did not: the two finer deltas run
--only-kinopaxplus, so at those deltas the CountingStars / CleanCost / KPAXCap / KPAX series do not
exist at all. The .sh says so with DELTA_EXTRA_ARGS, the .m says so with deltaPlusOnly, and they have
to agree or the plot expects series the sweep never wrote.

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
    mo = re.search(r'static const %s %s\[\]\s*=\s*\{([^}]*)\}' % (ctype, name), cu)
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


def m_str(name):
    mo = re.search(r"^%s\s*=\s*'([^']*)'" % name, m, re.M)
    if not mo:
        sys.exit("FATAL: %s = '...' not found in %s" % (name, M))
    return mo.group(1)


def m_bools(name):
    mo = re.search(r'^%s\s*=\s*\[([^\]]*)\]' % name, m, re.M)
    if not mo:
        sys.exit('FATAL: %s = [...] not found in %s' % (name, M))
    return [w == 'true' for w in re.findall(r'true|false', mo.group(1))]


def tok(x):
    """The label token convention: round(100 x float) for the fractional axes."""
    return int(round(100.0 * x))


# ---------------------------------------------------------------- the C++ side
cu_goalf   = cu_array('GOAL_FRONTIER_SIZES', 'int')
cu_efrac   = cu_array('EXPLORE_FRACS')
cu_kcap    = cu_array('KPAXCAP_CAPS')
cu_cap_derived = cu_scalar('CAP_DERIVED')
cu_dgf = cu_scalar('CS_DERIVED_GOAL_FRONTIER', 'int')
cu_def = cu_scalar('CS_DERIVED_EXPLORE_FRAC')

cu_clean = {}
for fld, key in (('CLEAN_BASE_W', 'w'), ('CLEAN_BASE_K', 'k'), ('CLEAN_BASE_CAP', 'cap')):
    cu_clean[key] = tok(cu_scalar(fld))
rmo = re.search(r'static const bool\s+CLEAN_BASE_R2\s*=\s*(true|false)', cu)
cu_clean['r2'] = 'on' if (rmo and rmo.group(1) == 'true') else 'off'

sh_deltas = sh_array('DELTA_LABELS')
sh_extra = sh_array('DELTA_EXTRA_ARGS')
# The quoted-string regex drops empty entries, so pad from the FRONT: index 0 is the full sweep and
# is the one with an empty flag string.
while len(sh_extra) < len(sh_deltas):
    sh_extra.insert(0, '')
sh_plus_only = ['--only-kinopaxplus' in e for e in sh_extra]
sh_metrics = sh_array('COST_LABELS')

problems = []

# --- Assertion 1: every derived point must be a member of its own list. --single-point selects BY
# VALUE, so a derived point outside the grid means that pass runs nothing at all.
for val, lst, a, b in ((cu_dgf, cu_goalf, 'CS_DERIVED_GOAL_FRONTIER', 'GOAL_FRONTIER_SIZES'),
                       (cu_def, cu_efrac, 'CS_DERIVED_EXPLORE_FRAC', 'EXPLORE_FRACS'),
                       (cu_cap_derived, cu_kcap, 'CAP_DERIVED', 'KPAXCAP_CAPS')):
    if not any(abs(v - val) < 1e-6 for v in lst):
        problems.append('%s (%g) is not in %s %s' % (a, val, b, lst))

# --- Assertion 2: the axes must stay in their meaningful ranges. goal_frontier_size is the NODE
# BUDGET for one iteration, so below 1 the frontier is empty and the search cannot advance;
# explore_frac is a SHARE of the remaining budget, so outside [0, 1] it is not a share at all.
if any(v < 1 for v in cu_goalf):
    problems.append('GOAL_FRONTIER_SIZES %s has an entry below 1 -- an empty frontier cannot '
                    'advance the search' % (cu_goalf,))
if any(v < 0.0 or v > 1.0 for v in cu_efrac):
    problems.append('EXPLORE_FRACS %s has an entry outside [0, 1] -- it is a share of the remaining '
                    'budget, not a count' % (cu_efrac,))


def cs_skip(goalf, efrac):
    """Mirrors countingStarsSkip(): FULL FACTORIAL, so --single-point is the only skip."""
    return False


cu_pairs = set()
for d, plus_only in zip(sh_deltas, sh_plus_only):
    if not plus_only:
        for goalf in cu_goalf:
            for efrac in cu_efrac:
                if cs_skip(goalf, efrac):
                    continue
                cu_pairs.add(('CountingStars_B%d_e%d'
                              % (int(round(goalf)), tok(efrac)), d))
        cu_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                      % (cu_clean['r2'], cu_clean['w'], cu_clean['k'], cu_clean['cap']), d))
        for c in cu_kcap:
            cu_pairs.add(('KPAXCap_cap%d' % tok(c), d))
        cu_pairs.add(('KPAX', d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_goalf   = m_ints('csGoalFrontierSizes')
m_efrac   = m_ints('csExploreFracs')
m_kcap    = m_ints('kpaxCapCaps')
m_deltas  = m_cellstr('deltas')
m_plus_only = m_bools('deltaPlusOnly')
m_dgf = m_scalar_int('csDerivedGoalFrontier')
m_def = m_scalar_int('csDerivedExploreFrac')
m_clean = {
    'r2': m_str('cleanBaseR2'),
    'w': m_scalar_int('cleanBaseW'),
    'k': m_scalar_int('cleanBaseK'),
    'cap': m_scalar_int('cleanBaseCap'),
}

m_pairs = set()
for d, plus_only in zip(m_deltas, m_plus_only):
    if not plus_only:
        for goalf in m_goalf:
            for efrac in m_efrac:
                m_pairs.add(('CountingStars_B%d_e%d' % (goalf, efrac), d))
        m_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                     % (m_clean['r2'], m_clean['w'], m_clean['k'], m_clean['cap']), d))
        for c in m_kcap:
            m_pairs.add(('KPAXCap_cap%d' % c, d))
        m_pairs.add(('KPAX', d))
    m_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- diff
only_cu = sorted(cu_pairs - m_pairs)
only_m = sorted(m_pairs - cu_pairs)

if sh_deltas != m_deltas:
    problems.append('DELTA_LABELS %s (%s) != deltas %s (%s)' % (sh_deltas, SH, m_deltas, M))
if sh_plus_only != m_plus_only:
    problems.append('--only-kinopaxplus flags %s (%s) != deltaPlusOnly %s (%s)'
                    % (sh_plus_only, SH, m_plus_only, M))

# --- Assertion 3: THE FILENAMES MUST MATCH END TO END.
#
# Matching label SETS is not enough, and this is the assertion that would have caught the two bugs
# that actually shipped. Both sides agreed perfectly on the label `CountingStars_r0_h1_e300` while:
#
#   * writePerIterationCSV()'s planner-name whitelist did not know it, so it fell through to the
#     KinoPaxPlus arm -- which keys on the DELTA and omits build_delta entirely. The length and
#     effort builds then wrote the SAME path, and the second overwrote the first.
#   * loadRuns()'s whitelist did not know it either, so the plot script error()d on it.
#
# So model both filename constructions from their real source, and diff the resulting PATHS.

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

# --- Assertion 4: distinct floats must not collapse onto the same label token. 0.01 and 0.1 both
# look plausible and both want "cap1"/"cap10"; a collision means two grid points silently write to
# ONE filename and the second overwrites the first.
for name, vals, f in (('EXPLORE_FRACS', cu_efrac, tok),
                      ('KPAXCAP_CAPS', cu_kcap, tok),
                      ('GOAL_FRONTIER_SIZES', cu_goalf, lambda v: int(round(v)))):
    toks = [f(v) for v in vals]
    if len(set(toks)) != len(set(vals)):
        problems.append('TOKEN COLLISION in %s: %s -> %s' % (name, vals, toks))

print('cost metrics : %s' % ', '.join(sh_metrics))
print('deltas       : %s  (--only-kinopaxplus: %s)'
      % (', '.join(sh_deltas), ', '.join(str(b) for b in sh_plus_only)))
print('series (.cu) : %d' % len(cu_pairs))
print('series (.m)  : %d' % len(m_pairs))

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
