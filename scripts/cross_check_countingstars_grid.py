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

v3 ADDS TWO CHECKS THAT COULD NOT EXIST IN v2. The budget now splits by three fixed fractions rather
than one share plus a remainder, so explore_frac + cost_frac can be oversubscribed -- and the planner
floors react_frac at 0 rather than failing, which makes it silent. And B is DERIVED from
MAX_TREE_SIZE and MAX_ITER rather than swept, so a fill_frac small enough to round B down to 0 is
also silent (the planner clamps to 1 and the run merely looks slow). Both are asserted below,
against constants parsed out of the .sh heredoc that writes config.h.

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

    fill_frac uses tok() (100x) instead: it is a coarse axis on {0.25, 0.5, 0.75} and `ff75` reads
    as three quarters where `ff750` does not.
    """
    return int(round(1000.0 * x))


# ---------------------------------------------------------------- the C++ side
cu_ffrac  = cu_array('FILL_FRACS')
cu_efrac  = cu_array('EXPLORE_FRACS')
cu_cfrac  = cu_array('COST_FRACS')
cu_blocks = cu_scalar('CS_MAX_BLOCKS', 'int')
cu_kcap   = cu_array('KPAXCAP_CAPS')
cu_cap_derived = cu_scalar('CAP_DERIVED')
cu_dff = cu_scalar('CS_DERIVED_FILL_FRAC')
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
for val, lst, a, b in ((cu_dff, cu_ffrac, 'CS_DERIVED_FILL_FRAC', 'FILL_FRACS'),
                       (cu_def, cu_efrac, 'CS_DERIVED_EXPLORE_FRAC', 'EXPLORE_FRACS'),
                       (cu_dcf, cu_cfrac, 'CS_DERIVED_COST_FRAC', 'COST_FRACS'),
                       (cu_cap_derived, cu_kcap, 'CAP_DERIVED', 'KPAXCAP_CAPS')):
    if not any(abs(v - val) < 1e-6 for v in lst):
        problems.append('%s (%g) is not in %s %s' % (a, val, b, lst))

# --- Assertion 2: the axes must stay in their meaningful ranges.
#
# fill_frac is the SHARE OF THE TREE BUFFER B is sized to consume per iteration, so outside (0, 1]
# it is not a share; at 0 the frontier is empty and the search cannot advance. explore_frac and
# cost_frac are SHARES OF B, so each must be in [0, 1] on its own.
if any(v <= 0.0 or v > 1.0 for v in cu_ffrac):
    problems.append('FILL_FRACS %s has an entry outside (0, 1] -- it is the share of the tree buffer '
                    'B consumes per iteration, and at 0 the frontier is empty' % (cu_ffrac,))
for name, vals in (('EXPLORE_FRACS', cu_efrac), ('COST_FRACS', cu_cfrac)):
    if any(v < 0.0 or v > 1.0 for v in vals):
        problems.append('%s %s has an entry outside [0, 1] -- it is a share of B, not a count'
                        % (name, vals))
if cu_blocks < 1:
    problems.append('CS_MAX_BLOCKS (%g) is below 1 -- rep >= 1 is a correctness floor' % cu_blocks)

# --- Assertion 2b: react_frac = 1 - explore_frac - cost_frac MUST STAY NON-NEGATIVE at every point
# on the grid. The planner floors it at 0, so an oversubscribed pair does not crash -- it silently
# switches the uniform DRAW off, and two grid points that differ only in how far past 1 they went
# would produce identical runs under different labels. NEW IN v3: v2 had one share and a remainder,
# so this could not be violated.
for ff in cu_ffrac:
    for ef in cu_efrac:
        for cf in cu_cfrac:
            if ef + cf > 1.0 + 1e-6:
                problems.append('OVERSUBSCRIBED BUDGET at (fill %g, explore %g, cost %g): '
                                'explore + cost = %g > 1, so react_frac would be negative and the '
                                'draw silently switches off' % (ff, ef, cf, ef + cf))

# --- Assertion 2c: the DERIVED B must be at least 1 at every fill_frac. B is
# floor(fill_frac * MAX_TREE_SIZE / MAX_ITER), and both constants are written into config.h by the
# .sh -- so this reads them from the heredoc rather than restating them. B < 1 means an empty
# frontier and a search that cannot advance, and it is silent: the planner clamps to 1 and the run
# looks merely slow.
cfg_tree = sh_config_int('MAX_TREE_SIZE')
cfg_iter = sh_config_int('MAX_ITER')
for ff in cu_ffrac:
    b = int(ff * cfg_tree / cfg_iter)
    if b < 1:
        problems.append('DERIVED B < 1 at fill_frac %g: floor(%g * %d / %d) = %d'
                        % (ff, ff, cfg_tree, cfg_iter, b))


def cs_label(ffrac, efrac, cfrac, blocks):
    """Mirrors countingStarsLabel() in the benchmark."""
    return 'CountingStars_ff%d_ef%d_cf%d_mb%d' % (tok(ffrac), ftok(efrac), ftok(cfrac),
                                                  int(round(blocks)))


cu_pairs = set()
for d, plus_only in zip(sh_deltas, sh_plus_only):
    if not plus_only:
        # FULL FACTORIAL: --single-point is the only skip, so there is no cs_skip() to mirror.
        for ffrac in cu_ffrac:
            for efrac in cu_efrac:
                for cfrac in cu_cfrac:
                    cu_pairs.add((cs_label(ffrac, efrac, cfrac, cu_blocks), d))
        cu_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                      % (cu_clean['r2'], cu_clean['w'], cu_clean['k'], cu_clean['cap']), d))
        for c in cu_kcap:
            cu_pairs.add(('KPAXCap_cap%d' % tok(c), d))
        cu_pairs.add(('KPAX', d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_ffrac   = m_ints('csFillFracs')
m_efrac   = m_ints('csExploreFracs')
m_cfrac   = m_ints('csCostFracs')
m_blocks  = m_ints('csMaxBlocks')
m_kcap    = m_ints('kpaxCapCaps')
m_deltas  = m_cellstr('deltas')
m_plus_only = m_bools('deltaPlusOnly')
m_dff = m_scalar_int('csDerivedFillFrac')
m_def = m_scalar_int('csDerivedExploreFrac')
m_dcf = m_scalar_int('csDerivedCostFrac')
m_dmb = m_scalar_int('csDerivedMaxBlocks')
m_clean = {
    'r2': m_str('cleanBaseR2'),
    'w': m_scalar_int('cleanBaseW'),
    'k': m_scalar_int('cleanBaseK'),
    'cap': m_scalar_int('cleanBaseCap'),
}

m_pairs = set()
for d, plus_only in zip(m_deltas, m_plus_only):
    if not plus_only:
        # The .m holds maxBlocks at csDerivedMaxBlocks rather than looping csMaxBlocks, exactly as
        # the .cu holds it at CS_MAX_BLOCKS -- so the single-entry list and the derived scalar have
        # to agree, which the derived-point diff below asserts.
        for ffrac in m_ffrac:
            for efrac in m_efrac:
                for cfrac in m_cfrac:
                    m_pairs.add(('CountingStars_ff%d_ef%d_cf%d_mb%d'
                                 % (ffrac, efrac, cfrac, m_dmb), d))
        m_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                     % (m_clean['r2'], m_clean['w'], m_clean['k'], m_clean['cap']), d))
        for c in m_kcap:
            m_pairs.add(('KPAXCap_cap%d' % c, d))
        m_pairs.add(('KPAX', d))
    m_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- diff
only_cu = sorted(cu_pairs - m_pairs)
only_m = sorted(m_pairs - cu_pairs)

if (tok(cu_dff), ftok(cu_def), ftok(cu_dcf), int(cu_blocks)) != (m_dff, m_def, m_dcf, m_dmb):
    problems.append('DERIVED POINT DRIFT: .cu (ff%d, ef%d, cf%d, mb%d) != .m (ff%d, ef%d, cf%d, mb%d)'
                    % (tok(cu_dff), ftok(cu_def), ftok(cu_dcf), int(cu_blocks),
                       m_dff, m_def, m_dcf, m_dmb))

# maxBlocks is held rather than swept on both sides, so the .m's one-entry list must name the value
# the .cu holds. A mismatch produces labels for a maxBlocks nothing ran.
if m_blocks != [int(cu_blocks)]:
    problems.append('csMaxBlocks %s (%s) does not match CS_MAX_BLOCKS %d (%s) -- maxBlocks is held, '
                    'not swept, so the list must be exactly the held value'
                    % (m_blocks, M, int(cu_blocks), CU))

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
for name, vals, f in (('FILL_FRACS', cu_ffrac, tok),
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
