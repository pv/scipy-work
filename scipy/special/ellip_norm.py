from . import ellip_normal
from ellip_normal import _ellipsoid_norm
import threading
import numpy as np

_ellip_lock = threading.Lock()

def ellip_normal(h2, k2, n, p):
    r"""
    Normalization constant for Ellipsoidal Harmonic Functions: the first kind

    .. math:: 

    \gamma^p_n=8\int_{0}^{h}\int_{h}^{k}\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}dydx

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
    \gamma^p_n(s) : double

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
    >>> w = ellip_normal(5,8,3,7)
    >>> w
    >>> 1723.38796997

    """
    return _ellipsoid_norm(h2, k2, n, p)
