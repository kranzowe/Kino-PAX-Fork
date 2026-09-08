#!/usr/bin/env python3
"""Assert the region decode math, and that CountingStars carries none of its own R2 machinery.

WHY THIS EXISTS. `getRegion` (src/graphs/Graph.cu) packs a state into an R1 index; a separate kernel
must unpack that index back into the region's minimum corner if anything needs `getSubRegion` to
measure a candidate's offset from that corner and find its R2 sub-region. If the two disagree, the
corner belongs to a different region than the index names, the offset is measured against the wrong
origin, and `getSubRegion`'s per-axis clamps quietly pin the result to an edge cell -- nothing fails,
the R2 identities are simply wrong.

`initializeRegions_kernel` in Graph.cu does disagree with `getRegion`'s encode, in two independent
ways:

  * DIGIT ORDER REVERSED. The encode makes wRegion the MOST significant group; the decode reads it
    from the LEAST significant end.
  * HARDCODED EXPONENTS. It uses C_R1_LENGTH**2 and V_R1_LENGTH**1 where the encode uses
    C_R1_LENGTH**C_DIM and V_R1_LENGTH**V_DIM.

The collapse factor is C_R1_LENGTH**(C_DIM-2) * V_R1_LENGTH**(V_DIM-1), so it is 8x at the
checked-in config and GROWS with a finer discretisation. This script measures that collapse rather
than asserting the number, so it stays true at any config. Graph.cu is deliberately NOT fixed: every
existing baseline (KPAX, KPAXCap, ...) was measured against it, and changing it would move them all.

COUNTINGSTARS USED TO CARRY A CORRECTED COPY of the decode above, purely to feed `getSubRegion` for
an R2 coverage diagnostic (r2_coverage_pct) that nothing plotted. Both are gone now: CountingStars
calls neither `getSubRegion` nor any min-corner table at all -- ordinality replaced novelty as the
freshness signal a whole planner version ago, and no door has read a sub-cell since. Check 5 below
now asserts that absence rather than the copy's correctness.

Run from anywhere:  python scripts/check_region_math.py
Exit 0 = REGION MATH OK, 1 = REGION MATH BROKEN.
"""
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, 'include', 'config', 'config.h')
CS = os.path.join(ROOT, 'src', 'planners', 'CountingStars.cu')


def cfg_int(name, text):
    mo = re.search(r'^#define\s+%s\s+(\d+)' % re.escape(name), text, re.M)
    if not mo:
        sys.exit('FATAL: %s not found in %s' % (name, CFG))
    return int(mo.group(1))


cfg = open(CFG, encoding='utf-8').read()
W_DIM = cfg_int('W_DIM', cfg)
C_DIM = cfg_int('C_DIM', cfg)
V_DIM = cfg_int('V_DIM', cfg)
WL = cfg_int('W_R1_LENGTH', cfg)
CL = cfg_int('C_R1_LENGTH', cfg)
VL = cfg_int('V_R1_LENGTH', cfg)

CPOW = CL ** C_DIM
VPOW = VL ** V_DIM
NUM_R1 = (WL ** W_DIM) * CPOW * VPOW

# getRegion has an early return for this degenerate case that ignores the attitude and velocity
# groups entirely, which would not be surjective onto NUM_R1_REGIONS. Not our config; refuse rather
# than silently validate a decode against an encode that cannot round-trip.
if V_DIM == 1 and C_DIM == 1:
    sys.exit('FATAL: getRegion takes its `V_DIM == 1 && C_DIM == 1` early return at this config; '
             'that path returns wRegion alone and is not a bijection. This checker does not model it.')


def group_encode(idx, length, dim):
    """Mirrors getRegion's per-group loop: axis 0 is the MOST significant digit.

    for(i = dim-1; i >= 0; --i) { region += factor * index[i]; factor *= length; }
    """
    region, factor = 0, 1
    for i in range(dim - 1, -1, -1):
        region += factor * idx[i]
        factor *= length
    return region


def group_decode(region, length, dim):
    """Inverse of group_encode. This half is already correct in Graph.cu."""
    idx = [0] * dim
    temp = region
    for i in range(dim - 1, -1, -1):
        idx[i] = temp % length
        temp //= length
    return idx


def encode(w_idx, a_idx, v_idx):
    """getRegion's return, in config terms:
       r1 = wRegion * CL**C_DIM * VL**V_DIM + aRegion * VL**V_DIM + vRegion
    """
    w = group_encode(w_idx, WL, W_DIM)
    a = group_encode(a_idx, CL, C_DIM) if CL > 1 else 0
    v = group_encode(v_idx, VL, V_DIM) if VL > 1 else 0
    return w * CPOW * VPOW + a * VPOW + v


def decode_fixed(r1):
    """THE CORRECTED DECODE -- what CountingStars must implement.

    Strip the groups in reverse significance: velocity is the least significant group, then
    attitude, and whatever remains is the workspace group.
    """
    v = r1 % VPOW
    a = (r1 // VPOW) % CPOW
    w = r1 // (VPOW * CPOW)
    return group_decode(w, WL, W_DIM), group_decode(a, CL, C_DIM), group_decode(v, VL, V_DIM)


def decode_current(tid):
    """What initializeRegions_kernel in Graph.cu actually does today -- reproduced to MEASURE the
    damage, not to be used. Note `CL*CL` and a bare `% VL`, both independent of C_DIM / V_DIM."""
    w = tid % (WL ** W_DIM)
    a = (tid // (WL ** W_DIM)) % (CL * CL)
    v = (tid // ((WL ** W_DIM) * CL * CL)) % VL
    return group_decode(w, WL, W_DIM), group_decode(a, CL, C_DIM), group_decode(v, VL, V_DIM)


problems = []

# --- 1. The corrected decode round-trips for every region. ---
bad = 0
for r1 in range(NUM_R1):
    w_idx, a_idx, v_idx = decode_fixed(r1)
    if encode(w_idx, a_idx, v_idx) != r1:
        bad += 1
        if bad <= 3:
            problems.append('decode_fixed(%d) -> encode = %d' % (r1, encode(w_idx, a_idx, v_idx)))
if bad:
    problems.append('corrected decode fails to round-trip on %d of %d regions' % (bad, NUM_R1))

# --- 2. It is a bijection: every region yields a DISTINCT corner. This is the property the buggy
# decode loses, and the one getSubRegion actually depends on. ---
fixed_corners = {(tuple(a), tuple(b), tuple(c))
                 for a, b, c in (decode_fixed(r1) for r1 in range(NUM_R1))}
if len(fixed_corners) != NUM_R1:
    problems.append('corrected decode is NOT a bijection: %d distinct corners for %d regions'
                    % (len(fixed_corners), NUM_R1))

# --- 3. Every axis index stays in range. ---
for r1 in (0, NUM_R1 // 2, NUM_R1 - 1):
    w_idx, a_idx, v_idx = decode_fixed(r1)
    for name, idx, length in (('w', w_idx, WL), ('a', a_idx, CL), ('v', v_idx, VL)):
        if any(not (0 <= k < length) for k in idx):
            problems.append('decode_fixed(%d) %s index out of range: %s (length %d)'
                            % (r1, name, idx, length))

# --- 4. Measure the current kernel's collapse, so the scoped fix is justified by measurement. ---
cur_corners = {(tuple(a), tuple(b), tuple(c))
               for a, b, c in (decode_current(r1) for r1 in range(NUM_R1))}
collapse = NUM_R1 / len(cur_corners) if cur_corners else 0
predicted = (CL ** max(0, C_DIM - 2)) * (VL ** max(0, V_DIM - 1))

# --- 5. If CountingStars exists, check it carries NO min-corner / R2 machinery of its own -- the
# dead-structure removal took getSubRegion, CountingStars_initializeRegions_kernel and d_minCornerCS_
# out entirely, so any of them reappearing means R2 tracking (or a broken copy of it) crept back in. ---
cs_note = ''
if os.path.exists(CS):
    cs = open(CS, encoding='utf-8').read()
    if 'getSubRegion' in cs:
        problems.append('CountingStars.cu calls getSubRegion -- it was removed along with the R2 '
                        'coverage pipeline; a call site means either that removal was incomplete or '
                        'R2 tracking is being reintroduced without a corrected min-corner table')
    if re.search(r'initializeRegions_kernel', cs):
        problems.append('CountingStars.cu references an initializeRegions_kernel (shared or its own '
                        '-- both are gone) -- if R2 tracking is being reintroduced it needs the '
                        'corrected decode back, not the shared broken one')
    if 'd_minCornerCS_' in cs or 'graph_.d_minValueInRegion_ptr_' in cs:
        problems.append('CountingStars.cu references a min-corner table -- neither its own corrected '
                        'one nor the shared broken one should exist now that nothing calls getSubRegion')
else:
    cs_note = '  (CountingStars.cu not present yet -- skipped check 5)'

print('config        : W %d^%d   C %d^%d   V %d^%d' % (WL, W_DIM, CL, C_DIM, VL, V_DIM))
print('NUM_R1_REGIONS: %d' % NUM_R1)
print('corrected     : %d distinct corners  (bijection: %s)'
      % (len(fixed_corners), 'yes' if len(fixed_corners) == NUM_R1 else 'NO'))
print('Graph.cu today: %d distinct corners  -> %.0fx collapse (predicted %.0fx)'
      % (len(cur_corners), collapse, predicted))
if cs_note:
    print(cs_note)

if problems:
    print('\nProblems (%d):' % len(problems))
    for p in problems:
        print('  ' + p)

print('\n%s' % ('REGION MATH OK' if not problems else 'REGION MATH BROKEN'))
sys.exit(0 if not problems else 1)
