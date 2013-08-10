"""
Convergence regions of the expansions used in ``hyp0f1.c``


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


def do_plot():
    from scipy.special._ufuncs import \
         _hyp_0f1_power_series

    bs = np.linspace(-10, 10, 131)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 20, 31)[1:]])

    rp = _hyp_0f1_power_series(bs[:,None], zs[None,:])

    mpmath.mp.dps = 50
    def sh(b, z):
        try:
            return float(mpmath.hyp0f1(mpmath.mpf(b), mpmath.mpf(z)))
        except ZeroDivisionError:
            return np.nan
    ex = np.vectorize(sh, otypes='d')(bs[:,None], zs[None,:])

    err_p = err_metric(rp[0], ex) + 1e-300

    err_est_p = abs(rp[1]/rp[0])

    levels = [-1000, -12]

    plt.cla()

    plt.hold(1)

    plt.contourf(bs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contour(bs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], linestyles=[':', ':'])
    lp = plt.contour(bs, zs, np.log10(err_est_p).T, levels=levels, colors=['r', 'r'], linestyles=['-', '-'])
    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})

    plt.xlim(bs.min(), bs.max())
    plt.ylim(zs.min(), zs.max())

    plt.xlabel('a')
    plt.ylabel('z')


def main():
    plt.clf()
    plt.subplot(121)
    do_plot()
    plt.title('0F1(b, z)')

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
