#!/usr/bin/env python3
"""Assert every CUDA kernel's declaration, definition and launch sites agree.

WHY THIS EXISTS. CUDA_SEPARABLE_COMPILATION is ON in this project and KPAX's kernels are UNPREFIXED
file-scope globals (`propagateFrontier_kernel1` and friends). So when a planner is derived by copy,
renaming the class does not rename the kernels -- and an unprefixed copy is a DUPLICATE-SYMBOL LINK
error, not a compile error. A clean compile proves nothing. The tuning sweep also pulls several
planner headers into ONE translation unit, so a file-scope `static const` that was not renamed is a
redefinition error in exactly the one .cu that matters and nowhere else.

There is no nvcc on the authoring machine, so this is the only pre-build check for either.

Checks, in order of how loudly they would otherwise fail:

  1. DUPLICATE DEFINITION   the same __global__ name defined in more than one .cu  -> LINK error
  2. ARITY MISMATCH         declaration vs definition parameter count
  3. LAUNCH ARITY           a `name<<<...>>>(args)` whose top-level argument count differs from the
                            definition's parameter count
  4. UNDECLARED / UNDEFINED a kernel launched or declared with no definition anywhere
  5. NO SHARED PREFIX /     a planner .cu whose kernels do not all share one file-specific prefix,
     PREFIX SHARED          or whose prefix another planner file also uses. The prefix need not be
                            the class name (KinoPaxPlusSH_, prune_ and original_ are all fine) --
                            it only has to be shared within the file and unique across files. This
                            is check 1 waiting to happen the next time the file is copied.

Checks 1-4 are ERRORS; check 5 is a warning and does not affect the exit code. KPAX.cu is expected
to appear under check 5 -- its unprefixed kernels are the reason all of this exists.

Run from anywhere:  python scripts/check_sigs.py
Exit 0 = SIGNATURES OK, 1 = SIGNATURES DIVERGE.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN_DIRS = ('src', 'include', 'examples')
EXTS = ('.cu', '.cuh')


def strip_comments(text):
    """Remove // and /* */ comments without disturbing line structure enough to matter."""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def split_top_level(s):
    """Split on commas that are not inside (), [], <> or a string/char literal."""
    out, depth, buf, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in '"\'':
            quote, buf_start = c, i
            i += 1
            while i < len(s) and s[i] != quote:
                i += 2 if s[i] == '\\' else 1
            buf.append(s[buf_start:i + 1])
            i += 1
            continue
        if c in '([{<':
            depth += 1
        elif c in ')]}>':
            depth -= 1
        if c == ',' and depth == 0:
            out.append(''.join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    tail = ''.join(buf)
    if tail.strip() or out:
        out.append(tail)
    return [p.strip() for p in out]


def arg_count(arglist):
    """Parameter/argument count for a parenthesised list body. `void` and empty both mean 0."""
    body = arglist.strip()
    if not body or body == 'void':
        return 0
    return len([p for p in split_top_level(body) if p.strip()])


def matching_paren(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, '/')


# ---------------------------------------------------------------- collect
files = []
for d in SCAN_DIRS:
    for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, d)):
        for fn in filenames:
            if fn.endswith(EXTS):
                files.append(os.path.join(dirpath, fn))
files.sort()

definitions = {}   # name -> list of (file, nparams)
declarations = {}  # name -> list of (file, nparams)
launches = {}      # name -> list of (file, nargs)

KERNEL_RE = re.compile(r'__global__\s+(?:static\s+)?void\s+([A-Za-z_]\w*)\s*\(')

for path in files:
    with open(path, encoding='utf-8', errors='replace') as f:
        raw = f.read()
    src = strip_comments(raw)

    for mo in KERNEL_RE.finditer(src):
        name = mo.group(1)
        open_idx = mo.end() - 1
        close_idx = matching_paren(src, open_idx)
        if close_idx < 0:
            continue
        n = arg_count(src[open_idx + 1:close_idx])
        after = src[close_idx + 1:close_idx + 40].lstrip()
        if after.startswith(';'):
            declarations.setdefault(name, []).append((path, n))
        else:
            definitions.setdefault(name, []).append((path, n))

    # Launch sites: name<<<grid, block>>>(args)
    for mo in re.finditer(r'([A-Za-z_]\w*)\s*<<<', src):
        name = mo.group(1)
        gt = src.find('>>>', mo.end())
        if gt < 0:
            continue
        open_idx = src.find('(', gt)
        if open_idx < 0:
            continue
        close_idx = matching_paren(src, open_idx)
        if close_idx < 0:
            continue
        launches.setdefault(name, []).append((path, arg_count(src[open_idx + 1:close_idx])))

# ---------------------------------------------------------------- checks
problems = []

# 1. Duplicate definitions ACROSS FILES -- the link error a clean compile will not catch. Two
# definitions in ONE file are a legal C++ overload (findInd takes bool* or uint*), not a collision.
for name, defs in sorted(definitions.items()):
    where = sorted({rel(p) for p, _ in defs})
    if len(where) > 1:
        problems.append('DUPLICATE DEFINITION  %s defined in %d files: %s'
                        % (name, len(where), ', '.join(where)))

# 2. Declaration vs definition arity. An overloaded kernel has several valid arities, so a
# declaration matching ANY definition is fine.
for name, decls in sorted(declarations.items()):
    if name not in definitions:
        continue
    valid = sorted({n for _p, n in definitions[name]})
    for path, n in decls:
        if n not in valid:
            problems.append('ARITY MISMATCH        %s: declaration in %s takes %d, definition takes %s'
                            % (name, rel(path), n, ' or '.join(str(v) for v in valid)))

# 3. Launch arity, against the same set.
for name, sites in sorted(launches.items()):
    if name not in definitions:
        continue
    valid = sorted({n for _p, n in definitions[name]})
    for path, n in sites:
        if n not in valid:
            problems.append('LAUNCH ARITY          %s: launch in %s passes %d, definition takes %s'
                            % (name, rel(path), n, ' or '.join(str(v) for v in valid)))

# 4. Launched or declared but never defined.
for name in sorted(set(launches) - set(definitions)):
    problems.append('NO DEFINITION         %s is launched (%s) but defined nowhere'
                    % (name, ', '.join(sorted({rel(p) for p, _ in launches[name]}))))
for name in sorted(set(declarations) - set(definitions)):
    problems.append('NO DEFINITION         %s is declared (%s) but defined nowhere'
                    % (name, ', '.join(sorted({rel(p) for p, _ in declarations[name]}))))

# 5. Planner files whose kernels do not share ONE file-specific prefix. The prefix need not be the
# class name -- KinoPaxPlusSpatialHash.cu uses KinoPaxPlusSH_ and PruneKPAX.cu uses prune_, both
# perfectly safe -- it only has to be shared by every kernel in the file and used by no other file.
# A file that fails this is what BECOMES check 1 the next time a planner is derived by copying it.
warnings = []
planner_dir = os.path.join(ROOT, 'src', 'planners')
SHARED_HELPERS = ('Planner', 'originalPlanner')   # findInd / repeatInd are meant to be unprefixed

by_file = {}
for name, defs in definitions.items():
    for path, _n in defs:
        if os.path.dirname(path) == planner_dir:
            by_file.setdefault(path, set()).add(name)


def first_token(name):
    return name.split('_', 1)[0] + '_' if '_' in name else None


file_prefix = {}
for path, names in by_file.items():
    toks = {first_token(n) for n in names}
    file_prefix[path] = toks.pop() if len(toks) == 1 and None not in toks else None

for path in sorted(by_file):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in SHARED_HELPERS:
        continue
    pfx = file_prefix[path]
    if pfx is None:
        warnings.append('NO SHARED PREFIX      %s: kernels %s do not share one prefix'
                        % (rel(path), ', '.join(sorted(by_file[path]))))
        continue
    others = [rel(q) for q in by_file
              if q != path and file_prefix.get(q) == pfx]
    if others:
        warnings.append('PREFIX SHARED         %s: prefix "%s" is also used by %s'
                        % (rel(path), pfx, ', '.join(sorted(others))))

print('files scanned    : %d' % len(files))
print('kernels defined  : %d' % len(definitions))
print('launch sites     : %d' % sum(len(v) for v in launches.values()))

if problems:
    print('\nProblems (%d):' % len(problems))
    for p in problems:
        print('  ' + p)

if warnings:
    print('\nUnprefixed kernels (%d) -- not an error today, a duplicate-symbol link error as soon as'
          '\nthe file is copied to derive another planner:' % len(warnings))
    for w in warnings:
        print('  ' + w)

print('\n%s' % ('SIGNATURES OK' if not problems else 'SIGNATURES DIVERGE'))
sys.exit(0 if not problems else 1)
