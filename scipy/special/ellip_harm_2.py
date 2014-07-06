from . import _ellip_harm_2
from _ellip_harm_2 import _ellipsoid
import threading
import numpy as np

_ellip_lock = threading.Lock()

def ellip_harm_2(h2, k2, n, p, s):
    r"""
    Ellipsoidal Harmonic functions F^p_n(l), also known as Lame Functions:The second kind

    .. math::

     F^p_n(s)=(2n + 1)E^p_n(s)\int_{0}^{1/s}\frac{du}{(E^p_n(1/u))^2\sqrt{(1-u^2k^2)(1-u^2h^2)}}

    Parameters
    ----------
    h2: double
        :math:`h^2`
    k2: double
        :math:`k^2`
    n: int
       degree
    p: int
       order, can range between [1,2n+1]
    
    Returns
    -------
    F^p_n(s) : double

    Notes
    -----
    The geometric intepretation is in accordance with [2],[3],[4]

    References
    ----------
    .. [1] Digital Libary of Mathematical Functions 29.12
       http://dlmf.nist.gov/29.12
    .. [2] Bardhan and Knepley.Computational science and 
       re-discovery: open-source implementations of 
       ellipsoidal harmonics for problems in potential theory
       http://arxiv.org/abs/1204.0267
    .. [3] G. Romain and B. Jean-Pierre. Ellipsoidal harmonic expansions
       of the gravitational potential: theory and applications.
       http://link.springer.com/article/10.1023%2FA%3A1017555515763#close
    .. [4] David J.and Dechambre P. Computation of Ellipsoidal
       Gravity Field Harmonics for small solar system bodies
       http://ccar.colorado.edu/scheeres/scheeres/assets/Theses%20and%20Abstracts/dechambre_thesis.pdf
    .. [5]George Dassios. Ellipsoidal Harmonics: Theory and Applications
    
    Examples
    --------
    >>> from scipy.special import ellip_harm_2
    >>> w = ellip_harm_2(5,8,2,1,10)
    >>> w
    >>> 0.00108056853382

    """
    return _ellipsoid(h2, k2, n, p, s)
