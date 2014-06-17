#
# Tests for the Ellipsoidal Harmonic Function,
# Adapted from the MPMath tests [1] by Yosef Meller, mellerf@netvision.net.il
# Distributed under the same license as SciPy itself.
#

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import ellip_harm
from numpy import nan, inf, pi, e, isnan, log, r_, array, sqrt, complex_

from scipy.special._testutils import FuncData


def test_values():
#    assert_equal(ellip_harm(3,2,1,1,2,3,1,1), lambertw(3,2,1,1,2,3))

    data = [
        (3,2,1,0,1,2.5,1,1, 1),
        (3,2,1,1,1,2.5,1,1, 2.5),
        (3,2,1,1,2,2.5,1,1, sqrt(abs(2.5 *2.5- (3*3-2*2)))),
        (3,2,1,1,3,2.5,1,1, sqrt(abs(2.5 *2.5- (3*3-1*1)))),
        (3,2,1,2,1,2.5,1,1, 2.5*2.5 - 1/3*((8+5)-sqrt(abs((8+5)*(8+5)-3*8*5)))),
        (3,2,1,2,2,2.5,1,1, 2.5*2.5 - 1/3*((8+5)+sqrt(abs((8+5)*(8+5)-3*8*5)))),
        (3,2,1,2,3,2.5,1,1, 2.5* sqrt(abs(6.25 - 5))),
        (3,2,1,2,4,2.5,1,1, 2.5* sqrt(abs(6.25 - 8))),
        (3,2,1,2,5,2.5,1,1, sqrt(abs((6.25 - 5)*(6.25 - 8)))),
        (3,2,1,3,2,2.5,1,1, 2.5*2.5*2.5 - 0.5*( 2*(8 + 5)+ sqrt(4*(8 + 5)*(8 + 5) - 15*5*8))),
        (3,2,1,3,3,2.5,1,1, sqrt(abs(6.25-5))*(6.25 - 1/5*((5 + 2*8)-sqrt(abs((2* 8+5)*(2*8+5)-5*8*5))))),
        (3,2,1,3,4,2.5,1,1, sqrt(abs(6.25-5))*(6.25 - 1/5*((5 + 2*8)+sqrt(abs((2* 8+5)*(2*8+5)-5*8*5))))),
        (3,2,1,3,5,2.5,1,1, sqrt(abs(6.25-8))*(6.25 - 1/5*((2*5 + 8)-sqrt(abs((2* 5+8)*(2*5+8)-5*8*5))))),
        (3,2,1,3,6,2.5,1,1, sqrt(abs(6.25-8))*(6.25 - 1/5*((2*5 + 8)+sqrt(abs((2* 5+8)*(2*5+8)-5*8*5))))),
        (3,2,1,3,7,2.5,1,1, 2.5 * sqrt(abs((6.25 - 5)*(6.25 - 8)))),
        ]

    data = array(data, dtype= float)

    def w(a, b, c, d, e, f, g, h):
        return ellip_harm(a, b, c, d, e, f, g, h)
    olderr = np.seterr(all='ignore')
    try:
        FuncData(w, data, (0,1,2,3,4,5,6,7), 8, rtol=1e-10, atol=1e-13).check()
        print ('hi')  
    finally:
        np.seterr(**olderr)

