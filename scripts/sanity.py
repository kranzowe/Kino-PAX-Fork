#!/usr/bin/env python3
"""Brace / paren / bracket balance for the CUDA sources.

WHY THIS EXISTS. There is no nvcc on the authoring machine, and the planner sources are routinely
edited by scripted block replacement rather than by hand. A replacement that drops or duplicates a
closing brace produces a file that looks fine in a diff and fails hundreds of lines later at the
next `}`. This catches that in a second, before anything is pushed to the machine that can compile.

It is a BALANCE check, not a parser: it knows about comments, string and character literals, and
raw-ish escapes, and nothing else. A balanced file is not a valid file -- run check_sigs.py too.

Usage:
    python scripts/sanity.py                 # every .cu / .cuh under src, include, examples
    python scripts/sanity.py path [path ...] # just these

Exit 0 = BALANCED, 1 = UNBALANCED.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN_DIRS = ('src', 'include', 'examples')
EXTS = ('.cu', '.cuh', '.h', '.hpp', '.cpp')

PAIRS = {'{': '}', '(': ')', '[': ']'}
CLOSERS = {v: k for k, v in PAIRS.items()}


def check(path):
    """Return a list of problem strings for one file."""
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()

    problems = []
    stack = []          # (char, line)
    line = 1
    i, n = 0, len(src)

    while i < n:
        c = src[i]

        if c == '\n':
            line += 1
            i += 1
            continue

        # --- comments ---
        if c == '/' and i + 1 < n:
            if src[i + 1] == '/':
                while i < n and src[i] != '\n':
                    i += 1
                continue
            if src[i + 1] == '*':
                end = src.find('*/', i + 2)
                if end < 0:
                    problems.append('%s: unterminated /* comment opened at line %d' % (rel(path), line))
                    return problems
                line += src.count('\n', i, end)
                i = end + 2
                continue

        # --- string / char literals ---
        if c in '"\'':
            quote, start_line = c, line
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == '\n':
                    line += 1          # a raw newline in a literal is its own problem; keep counting
                if src[i] == quote:
                    break
                i += 1
            if i >= n:
                problems.append('%s: unterminated %s literal opened at line %d'
                                % (rel(path), 'string' if quote == '"' else 'char', start_line))
                return problems
            i += 1
            continue

        # --- brackets. '<' is deliberately NOT tracked: it is an operator far more often than a
        # template bracket, and CUDA's <<< >>> would need special cases of its own. ---
        if c in PAIRS:
            stack.append((c, line))
        elif c in CLOSERS:
            if not stack:
                problems.append('%s:%d: stray %s with nothing open' % (rel(path), line, c))
            elif stack[-1][0] != CLOSERS[c]:
                opener, oline = stack[-1]
                problems.append('%s:%d: %s closes %s opened at line %d'
                                % (rel(path), line, c, opener, oline))
                stack.pop()
            else:
                stack.pop()
        i += 1

    for opener, oline in stack:
        problems.append('%s:%d: %s never closed' % (rel(path), oline, opener))
    return problems


def rel(path):
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, '/')
    except ValueError:
        return path


def collect():
    out = []
    for d in SCAN_DIRS:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, d)):
            for fn in filenames:
                if fn.endswith(EXTS):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def main(argv):
    targets = [os.path.abspath(a) for a in argv[1:]] or collect()
    problems = []
    for path in targets:
        problems.extend(check(path))

    print('files checked : %d' % len(targets))
    if problems:
        print('\nProblems (%d):' % len(problems))
        for p in problems:
            print('  ' + p)
    print('\n%s' % ('BALANCED' if not problems else 'UNBALANCED'))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
