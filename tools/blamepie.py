#!/usr/bin/env python
"""
blamepie.py

Generate a pie chart counting author source lines as reported by
git-blame, ignoring whitespace changes, binary files etc.

"""
from __future__ import print_function, division, absolute_import

import sys
import re
import subprocess
import argparse
import fnmatch

import numpy as np

INCLUDE_EXTENSIONS = ['.py', '.pyx', '.pxd',
                      '.c', '.h', '.cpp', '.hpp',
                      '.cxx', '.hxx', '.cc', '.hh',
                      '.f', '.f90', '.c.src', '.f.src']

EXTRA_IGNORE = [
    'scipy/fftpack/src/dfftpack/*.f',
    'scipy/fftpack/src/fftpack/*.f',
    'scipy/integrate/dop/*.f',
    'scipy/integrate/linpack_lite/*.f',
    'scipy/integrate/mach/*.f',
    'scipy/integrate/odepack/*.f',
    'scipy/integrate/quadpack/*.f',
    'scipy/interpolate/fitpack/*.f',
    'scipy/odr/odrpack/*.f',
    'scipy/optimize/cobyla/*.f',
    'scipy/optimize/lbfgsb/*.f',
    'scipy/optimize/minpack/*.f',
    'scipy/optimize/minpack2/*.f',
    'scipy/optimize/nnls/*.f',
    'scipy/optimize/slsqp/*.f',
    'scipy/optimize/tnc/*.c',
    'scipy/optimize/tnc/*.h',
    'scipy/sparse/tnc/*.c',
    'scipy/sparse/linalg/isolve/iterative/*.f.src',
    'scipy/sparse/linalg/dsolve/SuperLU/SRC/*.c',
    'scipy/sparse/linalg/dsolve/SuperLU/SRC/*.h',
    'scipy/sparse/linalg/dsolve/SuperLU/SRC/*.c',
    'scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/*.f',
    'scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/*.f',
    'scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/*.f',
    'scipy/spatial/qhull/src/*.c',
    'scipy/spatial/qhull/src/*.h',
    'scipy/special/Faddeeva.*',
    'scipy/special/amos/*.f',
    'scipy/special/cdflib/*.f',
    'scipy/special/cephes/*.c',
    'scipy/special/cephes/*.h',
    'scipy/special/mach/*.f',
    'scipy/special/specfun/*.f',
    'scipy/stats/statlib/*.f',
    'scipy/weave/blitz/blitz/*.h',
    'scipy/weave/blitz/blitz/*.cc',
    'scipy/weave/blitz/blitz/*.cpp',
    'scipy/weave/blitz/blitz/*/*.h',
    'scipy/weave/blitz/blitz/*/*.cc',
    'scipy/weave/blitz/blitz/*/*.cpp',
]

AUTHOR_ALIASES = {
    'rgommers': 'Ralf Gommers',
    'josef': 'Josef Perktold',
    'josef-pktd': 'Josef Perktold',
    'Travis E. Oliphant': 'Travis Oliphant',
}

def _is_extension_ok(fn):
    for ext in INCLUDE_EXTENSIONS:
        if fn.endswith(ext):
            return True
    return False

def _is_ignore_ok(fn):
    for pattern in EXTRA_IGNORE:
        if fnmatch.fnmatch(fn, pattern):
            return False
    return True

def is_filename_ok(fn):
    return _is_extension_ok(fn) and _is_ignore_ok(fn)

def main():
    p = argparse.ArgumentParser(usage=__doc__.strip())
    args = p.parse_args()

    # Grab a list of tracked non-binary files
    p = subprocess.Popen(['git', 'grep', '-I', '--name-only', '-e', ''],
                         stdout=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise ValueError("git failure!")
    file_list = [x.strip() for x in out.splitlines() if x.strip()]

    # Filter only approved extensions
    file_list = [fn for fn in file_list if is_filename_ok(fn)]

    # Generate blame output
    author_counts = {}

    for j, fn in enumerate(file_list):
        i = ((j + 1) * 40) // len(file_list)
        sys.stderr.write("\r[" + "."*i + " "*(40 - i) + "] " + fn[:35] + " "*max(0, 35 - len(fn)))
        sys.stderr.flush()

        p = subprocess.Popen(['git', 'blame', '-M', '-C', '-w', fn],
                             stdout=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise ValueError("git failure!")

        for line in out.splitlines():
            m = re.match(r'^[^(]*\((.*?)\d\d\d\d-\d\d-\d\d.*$', line.strip())
            author = m.group(1).strip()
            author = AUTHOR_ALIASES.get(author, author)
            if author in author_counts:
                author_counts[author] += 1
            else:
                author_counts[author] = 1

    sys.stderr.write("\n")

    # Sort by counts
    author_count_list = author_counts.items()
    author_count_list.sort(key=lambda x: (x[1], x[0]))

    authors = [x[0] for x in author_count_list]
    counts = [x[1] for x in author_count_list]

    # Save result
    with open('blamepie.txt', 'w') as f:
        for k, v in author_count_list:
            f.write("%d %s\n" % (v, k))

    # Produce plot
    counts = np.asarray(counts, dtype=int)
    authors = np.asarray(authors, dtype='S64')

    import matplotlib.pyplot as plt
    plt.gcf().set_size_inches(8, 8)
    plt.pie(counts, labels=authors)
    plt.axis('equal')
    plt.savefig('blamepie.png', dpi=90)

if __name__ == "__main__":
    main()
