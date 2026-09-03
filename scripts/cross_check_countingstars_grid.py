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

v3 ADDED A CHECK THAT COULD NOT EXIST IN v2. The budget splits by fixed fractions rather than one
share plus a remainder, so explore_frac + cost_frac can be oversubscribed -- and the planner floors
react_frac at 0 rather than failing, which makes it silent. Asserted below.

v3.2 CHANGES WHAT "B" MEANS, AND THE OLD "B < 1 IS BAD" ASSERTION WITH IT. B used to be a single
fill_frac, required strictly > 0 because 0 meant a permanently empty budget-driven frontier. B is
now a per-iteration RAMP -- B_frac(x) = bufferSlope*x + bufferFloor -- and (bufferSlope,
bufferFloor) = (0, 0) is an INTENTIONAL grid point (the deepest ablation arm: OPTIMAL and the
region-best GUARANTEE stay uncapped regardless of B, so the frontier is never actually empty). So
"does B round to 0" is no longer inherently a bug; what still matters is that neither axis goes
NEGATIVE, which the code's floor-at-1 clamp would silently mask into a positive B that looks fine.
That is asserted below in place of the old strict-positivity check.

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


def sh_config_int(name):
    """Read a #define out of the write_config heredoc in the .sh.

    B is DERIVED from MAX_TREE_SIZE and MAX_ITER, and both are written into config.h by this script
    rather than living in the repo's checked-in config -- so the only honest place to read them for
    the derived-B assertion is the heredoc that writes them.
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
    indistinguishable from a genuine share of 0. It is kept so a stale CSV from a 100x grid cannot
    be silently loaded as the wrong series.

    bufferSlope/bufferFloor use tok() (100x) instead, matching v3's fill_frac convention: both are
    coarse axes (slope up to 1.5, floor up to 0.2) and `bs150`/`bf20` read directly as 1.5/0.2 where
    `bs1500`/`bf200` would not.
    """
    return int(round(1000.0 * x))


# ---------------------------------------------------------------- the C++ side
cu_slope  = cu_array('BUFFER_SLOPES')
cu_floor  = cu_array('BUFFER_FLOORS')
cu_efrac  = cu_array('EXPLORE_FRACS')
cu_cfrac  = cu_array('COST_FRACS')
cu_kcap   = cu_array('KPAXCAP_CAPS')
cu_cap_derived = cu_scalar('CAP_DERIVED')
cu_dslope = cu_scalar('CS_DERIVED_BUFFER_SLOPE')
cu_dfloor = cu_scalar('CS_DERIVED_BUFFER_FLOOR')
cu_def = cu_scalar('CS_DERIVED_EXPLORE_FRAC')
cu_dcf = cu_scalar('CS_DERIVED_COST_FRAC')

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
for val, lst, a, b in ((cu_dslope, cu_slope, 'CS_DERIVED_BUFFER_SLOPE', 'BUFFER_SLOPES'),
                       (cu_dfloor, cu_floor, 'CS_DERIVED_BUFFER_FLOOR', 'BUFFER_FLOORS'),
                       (cu_def, cu_efrac, 'CS_DERIVED_EXPLORE_FRAC', 'EXPLORE_FRACS'),
                       (cu_dcf, cu_cfrac, 'CS_DERIVED_COST_FRAC', 'COST_FRACS'),
                       (cu_cap_derived, cu_kcap, 'CAP_DERIVED', 'KPAXCAP_CAPS')):
    if not any(abs(v - val) < 1e-6 for v in lst):
        problems.append('%s (%g) is not in %s %s' % (a, val, b, lst))

# --- Assertion 2: the axes must stay in their meaningful ranges.
#
# v3.2: BUFFER_SLOPES / BUFFER_FLOORS replace the old fill_frac, which was required strictly > 0
# because 0 meant a permanently empty budget-driven frontier. That no longer holds -- (slope,
# floor) = (0, 0) is now an INTENTIONAL grid point (see the module docstring), and B_frac can
# legitimately exceed 1 (slope + floor up to 1.7 on this grid). The invariant that actually matters
# now is just non-negativity: a negative slope or floor would let B go negative, which the code's
# floor-at-1 clamp would silently turn into a positive B that looks fine. explore_frac and cost_frac
# are SHARES OF B, so each must still be in [0, 1] on its own.
for name, vals in (('BUFFER_SLOPES', cu_slope), ('BUFFER_FLOORS', cu_floor)):
    if any(v < 0.0 for v in vals):
        problems.append('%s %s has a negative entry -- B_frac = slope*x + floor could go negative, '
                        'and the planner\'s floor-at-1 clamp would silently mask it' % (name, vals))
for name, vals in (('EXPLORE_FRACS', cu_efrac), ('COST_FRACS', cu_cfrac)):
    if any(v < 0.0 or v > 1.0 for v in vals):
        problems.append('%s %s has an entry outside [0, 1] -- it is a share of B, not a count'
                        % (name, vals))

# --- Assertion 2b: react_frac = 1 - explore_frac - cost_frac MUST STAY NON-NEGATIVE at every point
# on the grid. The planner floors it at 0, so an oversubscribed pair does not crash -- it silently
# switches the uniform DRAW off, and two grid points that differ only in how far past 1 they went
# would produce identical runs under different labels. Independent of the buffer axes, which never
# entered this arithmetic even under v3.
for ef in cu_efrac:
    for cf in cu_cfrac:
        if ef + cf > 1.0 + 1e-6:
            problems.append('OVERSUBSCRIBED BUDGET at (explore %g, cost %g): '
                            'explore + cost = %g > 1, so react_frac would be negative and the '
                            'draw silently switches off' % (ef, cf, ef + cf))

# --- Assertion 2c: informational only, not a "problems" check -- see the module docstring for why
# the old strict "B < 1 is bad" framing no longer applies (bufferFloor = 0 is now intentional). Logs
# the ramp's minimum (at x = 0, i.e. bufferFloor alone -- the true infimum since slope >= 0 on every
# swept combination) so a reader can see it without re-deriving it, using the same MAX_TREE_SIZE /
# MAX_ITER read out of the .sh heredoc that write the ramp's real denominator.
cfg_tree = sh_config_int('MAX_TREE_SIZE')
cfg_iter = sh_config_int('MAX_ITER')
ramp_min_info = ['floor(%g * %d / %d) = %d' % (fl, cfg_tree, cfg_iter, int(fl * cfg_tree / cfg_iter))
                 for fl in cu_floor]


def cs_label(slope, floor, efrac, cfrac):
    """Mirrors countingStarsLabel() in the benchmark."""
    return 'CountingStars_bs%d_bf%d_ef%d_cf%d' % (tok(slope), tok(floor), ftok(efrac), ftok(cfrac))


cu_pairs = set()
for d, plus_only in zip(sh_deltas, sh_plus_only):
    if not plus_only:
        # FULL FACTORIAL: --single-point is the only skip, so there is no cs_skip() to mirror.
        for slope in cu_slope:
            for floor in cu_floor:
                for efrac in cu_efrac:
                    for cfrac in cu_cfrac:
                        cu_pairs.add((cs_label(slope, floor, efrac, cfrac), d))
        cu_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                      % (cu_clean['r2'], cu_clean['w'], cu_clean['k'], cu_clean['cap']), d))
        for c in cu_kcap:
            cu_pairs.add(('KPAXCap_cap%d' % tok(c), d))
        cu_pairs.add(('KPAX', d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_slope   = m_ints('csBufferSlopes')
m_floor   = m_ints('csBufferFloors')
m_efrac   = m_ints('csExploreFracs')
m_cfrac   = m_ints('csCostFracs')
m_kcap    = m_ints('kpaxCapCaps')
m_deltas  = m_cellstr('deltas')
m_plus_only = m_bools('deltaPlusOnly')
m_dslope = m_scalar_int('csDerivedBufferSlope')
m_dfloor = m_scalar_int('csDerivedBufferFloor')
m_def = m_scalar_int('csDerivedExploreFrac')
m_dcf = m_scalar_int('csDerivedCostFrac')
m_clean = {
    'r2': m_str('cleanBaseR2'),
    'w': m_scalar_int('cleanBaseW'),
    'k': m_scalar_int('cleanBaseK'),
    'cap': m_scalar_int('cleanBaseCap'),
}

m_pairs = set()
for d, plus_only in zip(m_deltas, m_plus_only):
    if not plus_only:
        for slope in m_slope:
            for floor in m_floor:
                for efrac in m_efrac:
                    for cfrac in m_cfrac:
                        m_pairs.add(('CountingStars_bs%d_bf%d_ef%d_cf%d'
                                     % (slope, floor, efrac, cfrac), d))
        m_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                     % (m_clean['r2'], m_clean['w'], m_clean['k'], m_clean['cap']), d))
        for c in m_kcap:
            m_pairs.add(('KPAXCap_cap%d' % c, d))
        m_pairs.add(('KPAX', d))
    m_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- diff
only_cu = sorted(cu_pairs - m_pairs)
only_m = sorted(m_pairs - cu_pairs)

if (tok(cu_dslope), tok(cu_dfloor), ftok(cu_def), ftok(cu_dcf)) \
        != (m_dslope, m_dfloor, m_def, m_dcf):
    problems.append('DERIVED POINT DRIFT: .cu (bs%d, bf%d, ef%d, cf%d) != '
                    '.m (bs%d, bf%d, ef%d, cf%d)'
                    % (tok(cu_dslope), tok(cu_dfloor), ftok(cu_def), ftok(cu_dcf),
                       m_dslope, m_dfloor, m_def, m_dcf))

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
for name, vals, f in (('BUFFER_SLOPES', cu_slope, tok),
                      ('BUFFER_FLOORS', cu_floor, tok),
                      ('EXPLORE_FRACS', cu_efrac, ftok),
                      ('COST_FRACS', cu_cfrac, ftok),
                      ('KPAXCAP_CAPS', cu_kcap, tok)):
    toks = [f(v) for v in vals]
    if len(set(toks)) != len(set(vals)):
        problems.append('TOKEN COLLISION in %s: %s -> %s' % (name, vals, toks))

print('cost metrics : %s' % ', '.join(sh_metrics))
print('deltas       : %s  (--only-kinopaxplus: %s)'
      % (', '.join(sh_deltas), ', '.join(str(b) for b in sh_plus_only)))
print('series (.cu) : %d' % len(cu_pairs))
print('series (.m)  : %d' % len(m_pairs))
print('ramp minimum : B(x=0) at each bufferFloor -- %s' % ', '.join(ramp_min_info))

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
