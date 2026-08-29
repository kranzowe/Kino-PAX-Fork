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
cu_react   = cu_array('REACT_COUNTS')
cu_half    = cu_array('FAN_HALF_LIVES', 'int')
cu_explore = cu_array('EXPLORE_COUNTS')
cu_kcap    = cu_array('KPAXCAP_CAPS')
cu_cap_derived = cu_scalar('CAP_DERIVED')
cu_dr = cu_scalar('CS_DERIVED_REACT')
cu_dh = cu_scalar('CS_DERIVED_HALFLIFE', 'int')
cu_de = cu_scalar('CS_DERIVED_EXPLORE')

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
for val, lst, a, b in ((cu_dr, cu_react, 'CS_DERIVED_REACT', 'REACT_COUNTS'),
                       (cu_dh, cu_half, 'CS_DERIVED_HALFLIFE', 'FAN_HALF_LIVES'),
                       (cu_de, cu_explore, 'CS_DERIVED_EXPLORE', 'EXPLORE_COUNTS'),
                       (cu_cap_derived, cu_kcap, 'CAP_DERIVED', 'KPAXCAP_CAPS')):
    if not any(abs(v - val) < 1e-6 for v in lst):
        problems.append('%s (%g) is not in %s %s' % (a, val, b, lst))

# --- Assertion 2: react = 0 must survive. It is the arm where the frontier is exactly this
# iteration's admissions -- the most KinoPaxPlus-like setting, and the only direct test of whether a
# small frontier is what actually matters. Losing it would gut the headline comparison.
if not any(abs(v) < 1e-9 for v in cu_react):
    problems.append('REACT_COUNTS %s has no 0 entry -- the smallest-frontier arm is missing'
                    % (cu_react,))


def cs_skip(react, half, explore):
    """Mirrors countingStarsSkip() in the benchmark: a CROSS, not a full factorial."""
    on_axis = abs(react - cu_dr) < 1e-6 and abs(half - cu_dh) < 1e-6
    at_derived = abs(explore - cu_de) < 1e-6
    return not at_derived and not on_axis


cu_pairs = set()
for d, plus_only in zip(sh_deltas, sh_plus_only):
    if not plus_only:
        for react in cu_react:
            for half in cu_half:
                for explore in cu_explore:
                    if cs_skip(react, half, explore):
                        continue
                    cu_pairs.add(('CountingStars_r%d_h%d_e%d'
                                  % (int(round(react)), int(round(half)), tok(explore)), d))
        cu_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                      % (cu_clean['r2'], cu_clean['w'], cu_clean['k'], cu_clean['cap']), d))
        for c in cu_kcap:
            cu_pairs.add(('KPAXCap_cap%d' % tok(c), d))
        cu_pairs.add(('KPAX', d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_react   = m_ints('csReactCounts')
m_half    = m_ints('csHalfLives')
m_explore = m_ints('csExploreCounts')
m_kcap    = m_ints('kpaxCapCaps')
m_deltas  = m_cellstr('deltas')
m_plus_only = m_bools('deltaPlusOnly')
m_dr = m_scalar_int('csDerivedReact')
m_dh = m_scalar_int('csDerivedHalfLife')
m_de = m_scalar_int('csDerivedExplore')
m_clean = {
    'r2': m_str('cleanBaseR2'),
    'w': m_scalar_int('cleanBaseW'),
    'k': m_scalar_int('cleanBaseK'),
    'cap': m_scalar_int('cleanBaseCap'),
}

m_pairs = set()
for d, plus_only in zip(m_deltas, m_plus_only):
    if not plus_only:
        for react in m_react:
            for half in m_half:
                for explore in m_explore:
                    on_axis = (react == m_dr and half == m_dh)
                    at_derived = (explore == m_de)
                    if not at_derived and not on_axis:
                        continue
                    m_pairs.add(('CountingStars_r%d_h%d_e%d' % (react, half, explore), d))
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

# --- Assertion 3: distinct floats must not collapse onto the same label token. 0.01 and 0.1 both
# look plausible and both want "cap1"/"cap10"; a collision means two grid points silently write to
# ONE filename and the second overwrites the first.
for name, vals, f in (('EXPLORE_COUNTS', cu_explore, tok),
                      ('KPAXCAP_CAPS', cu_kcap, tok),
                      ('REACT_COUNTS', cu_react, lambda v: int(round(v)))):
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
