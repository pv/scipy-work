"""
Continuous to discrete transformations for state-space and transfer function.
"""

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# March 29, 2011

import numpy as np
import numpy.linalg
import scipy.linalg

from ltisys import tf2ss, ss2tf

def _mrdivide(b,a):
    """Convenience function for matrix divides"""
    s = np.linalg.solve(a.transpose(), b.transpose())
    return s.transpose()

def ss_cont2discrete(a, b, c, d, dt, method="zoh"):
    """Transform a continuous to a discrete state-space system.

    The function defaults to a bilinear transform.

    Parameters
    -----------
    a, b, c, d : ndarray
        Arrays representing the continuous state-space system.
    dt : float
        The discretization time step.
    method : {"bilinear", "zoh"}
        Which method to use, bilinear or zero-order hold ("zoh", the default).

    Returns
    -------
    ad, bd, cd, dd : ndarray
        The equivalent discrete state-space system

    See Also
    --------
    tf_cont2discrete

    Notes
    -----
    By default, the routine uses a Zero-Order Hold (zoh) method
    to perform the transformation.  Alternatively, Tustin's
    bilinear approximation can be used.

    """
    if method=='bilinear':
        itv = 2.0/ dt * np.eye(a.shape[0])
        ad = _mrdivide((itv+a), (itv-a))
        iab = np.linalg.solve((itv-a), b)
        tk = 2.0 / dt
        bd = tk * iab
        cd = 2.0 * _mrdivide(c, (itv-a))
        dd = d + np.dot(c, iab)
    elif method=='zoh':
        em = np.vstack((np.hstack((a, b)),
                        np.hstack((np.zeros((b.shape[1], a.shape[1])),
                        np.zeros((b.shape[1], b.shape[1])) )) ))
        ms = scipy.linalg.expm(dt * em)
        ms = ms[:a.shape[0]]
        ad = ms[:, :a.shape[1]]
        bd = ms[:, a.shape[1]:]
        cd = c
        dd = d
    else:
        raise ValueError("Unknown transformation method.")

    return ad, bd, cd, dd

def tf_cont2discrete(num, den, dt, method="zoh"):
    """Transform a continuous to a discrete transfer function.

    The function defaults to a bilinear transform.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the numerator and denominator polynomials.
    dt : float
        The discretization time step.
    method : {"bilinear", "zoh"}
        Which method to use, bilinear or zero-order hold ("zoh", the default).

    Returns
    -------
    dnum, dden : array_like
        Sequences representing the numerator and denominator polynomials of the
        discrete transfer function.

    See Also
    --------
    ss_cont2discrete

    """
    a, b, c, d = tf2ss(num, den)
    ad, bd, cd, dd = ss_cont2discrete(a, b, c, d, dt, method=method)
    return ss2tf(ad, bd, cd, dd)
