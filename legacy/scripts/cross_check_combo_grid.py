#!/usr/bin/env python3
"""Assert the COMBO sweep's three files agree on every series label.

    examples/gpu/kinopaxstar_combo_tuning_sweep.cu   writes the CSVs
    scripts/run_combo_tuning_sweep.sh                chooses the deltas and per-delta flags
    scripts/process_combo_tuning_and_plot.m          decides which CSVs to load

WHY THIS EXISTS. When these drift, MATLAB does not error -- loadRuns() silently finds no files and
reports "0 runs" for the orphaned series, so the plot comes out looking merely sparse. That failure
mode has cost whole sweeps. Everything below is PARSED from the three real files; nothing is
restated by hand, because a hand-restated grid is just a fourth thing to drift.

Run from anywhere:  python scripts/cross_check_combo_grid.py
Exit 0 = GRIDS MATCH, 1 = GRIDS DIVERGE.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CU = os.path.join(ROOT, 'examples', 'gpu', 'kinopaxstar_combo_tuning_sweep.cu')
SH = os.path.join(ROOT, 'scripts', 'run_combo_tuning_sweep.sh')
M = os.path.join(ROOT, 'scripts', 'process_combo_tuning_and_plot.m')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


cu, sh, m = read(CU), read(SH), read(M)


# ---------------------------------------------------------------- parsers
def cu_floats(name):
    mo = re.search(r'static const float %s\[\]\s*=\s*\{([^}]*)\}' % name, cu)
    if not mo:
        sys.exit('FATAL: %s[] not found in %s' % (name, CU))
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', mo.group(1))]


def cu_scalar(name):
    mo = re.search(r'static const float %s\s*=\s*(-?\d+\.?\d*)f?' % name, cu)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CU))
    return float(mo.group(1))


def sh_array(name):
    """Anchored at ^ so the commented-out full variants in the .sh are correctly ignored."""
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
    """The label token convention shared by all three files: round(100 x float)."""
    return int(round(100.0 * x))


# ---------------------------------------------------------------- the C++ side
# COMBO's grid is sigma-multiple x fan-out-gain. A node is favoured when its fan-out score exceeds
# mu + N*sigma over the realised frontier's score distribution. kFan = 0 is the uniform control arm:
# every score is identical, sigma is 0, and the planner's degenerate branch gives every frontier node
# the same block count.
cu_sn = cu_floats('FAN_SIGMA_N')
cu_fan = cu_floats('FAN_GAINS')
cu_acc_gain = cu_scalar('ACC_GAIN')
cu_rf = cu_scalar('REACT_FRAC')
cu_true = cu_floats('TRUE_CAPS')
cu_kcap = cu_floats('KPAXCAP_CAPS')
cu_cap_derived = cu_scalar('CAP_DERIVED')
cu_sn_derived = cu_scalar('COMBO_DERIVED_SIGMA_N')
cu_fan_derived = cu_scalar('COMBO_DERIVED_FAN')

cu_clean = {}
for fld, key in (('CLEAN_BASE_W', 'w'), ('CLEAN_BASE_K', 'k'), ('CLEAN_BASE_CAP', 'cap')):
    cu_clean[key] = tok(cu_scalar(fld))
rmo = re.search(r'static const bool\s+CLEAN_BASE_R2\s*=\s*(true|false)', cu)
cu_clean['r2'] = 'on' if (rmo and rmo.group(1) == 'true') else 'off'

sh_deltas = sh_array('DELTA_LABELS')
sh_extra = sh_array('DELTA_EXTRA_ARGS')
# The quoted-string regex drops empty entries, so pad from the FRONT: the full-sweep deltas come
# first and are the ones with an empty flag string.
while len(sh_extra) < len(sh_deltas):
    sh_extra.insert(0, '')
sh_single = ['--single-point' in e for e in sh_extra]
sh_metrics = sh_array('COST_LABELS')

problems = []

# --- Assertion 1: every derived point must be a member of its own list. --single-point selects BY
# VALUE, so a derived point outside the grid means that pass runs nothing at all.
if not any(abs(c - cu_cap_derived) < 1e-6 for c in cu_true):
    problems.append('CAP_DERIVED (%g) is not in TRUE_CAPS %s' % (cu_cap_derived, cu_true))
if not any(abs(c - cu_cap_derived) < 1e-6 for c in cu_kcap):
    problems.append('CAP_DERIVED (%g) is not in KPAXCAP_CAPS %s' % (cu_cap_derived, cu_kcap))
if not any(abs(g - cu_sn_derived) < 1e-6 for g in cu_sn):
    problems.append('COMBO_DERIVED_SIGMA_N (%g) is not in FAN_SIGMA_N %s' % (cu_sn_derived, cu_sn))
if not any(abs(g - cu_fan_derived) < 1e-6 for g in cu_fan):
    problems.append('COMBO_DERIVED_FAN (%g) is not in FAN_GAINS %s' % (cu_fan_derived, cu_fan))
# kFan = 0 makes every fan-out score identical, so the planner hands every frontier node the same
# block count. It is the only direct test of whether sparse fan-out beats spreading it, so losing it
# would gut the headline comparison.
if not any(abs(g) < 1e-9 for g in cu_fan):
    problems.append('FAN_GAINS %s has no 0.0 entry -- the uniform control arm is missing' % (cu_fan,))
# N is measured in sigma, so 0 is meaningful (threshold at the mean) but not required -- that arm
# was swept, lost, and is documented rather than re-run every pass. What IS required is that the
# grid stay positive: a negative N would put the threshold BELOW the mean and favour the majority by
# construction, which is the one setting known to be wrong.
if any(g < 0.0 for g in cu_sn):
    problems.append('FAN_SIGMA_N %s has a negative entry -- that favours the majority by construction'
                    % (cu_sn,))


def combo_skip(sigmaN, kFan, single):
    """Mirrors comboSkip() in the benchmark.

    At kFan = 0 sigma is 0 and every N runs the identical uniform control arm, so only the derived N
    is kept there. Otherwise full factorial, and --single-point is the only other skip.
    """
    if abs(kFan) <= 1e-6 and abs(sigmaN - cu_sn_derived) > 1e-6:
        return True
    if not single:
        return False
    return abs(sigmaN - cu_sn_derived) > 1e-6 or abs(kFan - cu_fan_derived) > 1e-6


cu_pairs = set()
for d, single in zip(sh_deltas, sh_single):
    for sigmaN in cu_sn:
        for kFan in cu_fan:
            if combo_skip(sigmaN, kFan, single):
                continue
            cu_pairs.add(('KinoPaxSTARCOMBO_sn%d_kf%d_ka%d'
                          % (tok(sigmaN), tok(kFan), tok(cu_acc_gain)), d))
    cu_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                  % (cu_clean['r2'], cu_clean['w'], cu_clean['k'], cu_clean['cap']), d))
    for c in cu_true:
        if single and abs(c - cu_cap_derived) > 1e-6:
            continue
        cu_pairs.add(('KinoPaxSTARTrue_cap%d' % tok(c), d))
    for c in cu_kcap:
        if single and abs(c - cu_cap_derived) > 1e-6:
            continue
        cu_pairs.add(('KPAXCap_cap%d' % tok(c), d))
    cu_pairs.add(('KPAX', d))
    cu_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- the MATLAB side
m_sn = m_ints('comboFanSigmaN')
m_fan = m_ints('comboFanGains')
m_acc_gain = m_scalar_int('comboAccGain')
m_rf = m_scalar_int('comboReact')
m_true = m_ints('trueCaps')
m_kcap = m_ints('kpaxCapCaps')
m_deltas = m_cellstr('deltas')
m_single = m_bools('deltaSingleCap')
m_cap_derived = m_scalar_int('capDerived')
m_sn_derived = m_scalar_int('comboDerivedSigmaN')
m_fan_derived = m_scalar_int('comboDerivedFan')
m_clean = {
    'r2': m_str('cleanBaseR2'),
    'w': m_scalar_int('cleanBaseW'),
    'k': m_scalar_int('cleanBaseK'),
    'cap': m_scalar_int('cleanBaseCap'),
}

m_pairs = set()
for d, single in zip(m_deltas, m_single):
    for sigmaN in m_sn:
        for kFan in m_fan:
            if kFan == 0 and sigmaN != m_sn_derived:
                continue
            if single and not (sigmaN == m_sn_derived and kFan == m_fan_derived):
                continue
            m_pairs.add(('KinoPaxSTARCOMBO_sn%d_kf%d_ka%d' % (sigmaN, kFan, m_acc_gain), d))
    m_pairs.add(('KinoPaxSTARCleanCost_r2%s_w%d_k%d_cap%d'
                 % (m_clean['r2'], m_clean['w'], m_clean['k'], m_clean['cap']), d))
    for c in m_true:
        if single and c != m_cap_derived:
            continue
        m_pairs.add(('KinoPaxSTARTrue_cap%d' % c, d))
    for c in m_kcap:
        if single and c != m_cap_derived:
            continue
        m_pairs.add(('KPAXCap_cap%d' % c, d))
    m_pairs.add(('KPAX', d))
    m_pairs.add(('KinoPaxPlus', d))

# ---------------------------------------------------------------- diff
only_cu = sorted(cu_pairs - m_pairs)
only_m = sorted(m_pairs - cu_pairs)

if sh_deltas != m_deltas:
    problems.append('DELTA_LABELS %s (%s) != deltas %s (%s)' % (sh_deltas, SH, m_deltas, M))
if sh_single != m_single:
    problems.append('--single-point flags %s (%s) != deltaSingleCap %s (%s)'
                    % (sh_single, SH, m_single, M))

# --- Assertion 2: distinct floats must not collapse onto the same round(100x) token.
# 0.01 and 0.1 both look plausible and both want "cap1"/"cap10"; a collision here means two grid
# points silently write to ONE filename and the second overwrites the first.
for name, vals in (('FAN_SIGMA_N', cu_sn), ('FAN_GAINS', cu_fan),
                   ('TRUE_CAPS', cu_true), ('KPAXCAP_CAPS', cu_kcap)):
    toks = [tok(v) for v in vals]
    if len(set(toks)) != len(set(vals)):
        problems.append('TOKEN COLLISION in %s: %s -> %s' % (name, vals, toks))

print('cost metrics : %s' % ', '.join(sh_metrics))
print('deltas       : %s  (--single-point: %s)'
      % (', '.join(sh_deltas), ', '.join(str(b) for b in sh_single)))
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
