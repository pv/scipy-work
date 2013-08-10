"""
Convergence regions of the expansions used in ``hyp1f2.c``


Figure legend
=============

Red region
    Power series is close (1e-12) to the mpmath result

Dotted colored lines
    Boundaries of the regions

Solid colored lines
    Boundaries estimated by the routine itself. These will be used
    for determining which of the results to use.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

try:
    import mpmath
except:
    from sympy import mpmath


def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m


def do_plot(b1s, b2s):
    from scipy.special._ufuncs import \
         _hyp_1f2_power_series

    avs = np.linspace(-1000, 1000, 31)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 2000, 31)[1:]])

    rp = _hyp_1f2_power_series(avs[:,None], b1s, b2s, zs[None,:])

    mpmath.mp.dps = 50
    sh = lambda a, b1, b2, z: float(mpmath.hyp1f2(mpmath.mpf(a), mpmath.mpf(b1), mpmath.mpf(b2), mpmath.mpf(z)))
    ex = np.vectorize(sh, otypes='d')(avs[:,None], b1s, b2s, zs[None,:])

    err_p = err_metric(rp[0], ex) + 1e-300

    err_est_p = abs(rp[1]/rp[0])

    levels = [-1000, -12]

    plt.cla()

    plt.hold(1)

    plt.contourf(avs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contour(avs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], linestyles=[':', ':'])
    lp = plt.contour(avs, zs, np.log10(err_est_p).T, levels=levels, colors=['r', 'r'], linestyles=['-', '-'])
    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})

    plt.xlim(avs.min(), avs.max())
    plt.ylim(zs.min(), zs.max())

    plt.xlabel('a')
    plt.ylabel('z')


def main():
    plt.clf()
    plt.subplot(121)
    do_plot(1, -1.5)
    plt.title('1F2(a, 1, -1.5, z)')

    plt.savefig('struve_convergence.png')
    plt.show()

if __name__ == "__main__":
    import os, sys
    if '--main' in sys.argv:
        main()
    else:
        import subprocess
        subprocess.call([sys.executable, os.path.join('..', '..', '..', 'runtests.py'),
                         '-g', '--python', __file__, '--main'])
