import numpy as np
from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal

from scipy.signal import ss_cont2discrete as ss_c2d
from scipy.signal import tf_cont2discrete as tf_c2d

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# March 29, 2011

A_CONT = np.eye(2)
B_CONT = 0.5*np.ones((2,1))
C_CONT = np.array([[0.75,1.0],[1.0,1.0],[1.0,0.25]])
D_CONT = np.array([[0.0,],[0.0,],[-0.33,]])

DT_DISC = 0.5
A_DISC = 1.648721270700128*np.eye(2)
B_DISC = 0.324360635350064*np.ones((2,1))
C_DISC = C_CONT
D_DISC = D_CONT

A_DISC_BL = 1.666666666666667*np.eye(2)
B_DISC_BL = 0.666666666666667*np.ones((2,1))
C_DISC_BL = np.array([[0.5,2.0/3.0],[2.0/3.0,2.0/3.0],[2.0/3.0,1.0/6.0]])
D_DISC_BL = np.array([[0.291666666666667,],[1.0/3.0,],[-0.121666666666667,]])

DT_DISC_2 = 0.25
A_DISC_BL_2 = 1.285714285714286*np.eye(2)
B_DISC_BL_2 = 0.571428571428571*np.ones((2,1))
C_DISC_BL_2 = np.array([[0.214285714285714,0.285714285714286],[0.285714285714286,0.285714285714286],[0.285714285714286,0.071428571428571]])
D_DISC_BL_2 = np.array([[0.125,],[0.142857142857143,],[-0.240714285714286,]])

DT_DISC_3 = 1.0/3.0
A_DISC_BL_3 = 1.4*np.eye(2)
B_DISC_BL_3 = 0.6*np.ones((2,1))
C_DISC_BL_3 = np.array([[0.3,0.4],[0.4,0.4],[0.4,0.1]])
D_DISC_BL_3 = np.array([[0.175,],[0.2,],[-0.205,]])

NUM_CONT = np.array([0.25,0.25,0.5])
DEN_CONT = np.array([0.75,0.75,1.0])

# Uses dt=0.5
NUM_DISC = np.array([[1.0/3.0,-0.427419169438754,0.221654141101125],])
DEN_DISC = np.array([1.0,-1.351394049721225,0.606530659712634])

class TestC2D(TestCase):
    def test_zoh(self):

        ad,bd,cd,dd = ss_c2d(A_CONT,B_CONT,C_CONT,D_CONT,DT_DISC,method='zoh')

        assert_array_almost_equal(A_DISC,ad)
        assert_array_almost_equal(B_DISC,bd)
        assert_array_almost_equal(C_DISC,cd)
        assert_array_almost_equal(D_DISC,dd)

    def test_bilinear(self):
        ad,bd,cd,dd = ss_c2d(A_CONT, B_CONT, C_CONT, D_CONT, DT_DISC,
                             method='bilinear')

        assert_array_almost_equal(A_DISC_BL,ad)
        assert_array_almost_equal(B_DISC_BL,bd)
        assert_array_almost_equal(C_DISC_BL,cd)
        assert_array_almost_equal(D_DISC_BL,dd)

        ad,bd,cd,dd = ss_c2d(A_CONT, B_CONT, C_CONT, D_CONT, DT_DISC_2,
                             method='bilinear')

        assert_array_almost_equal(A_DISC_BL_2,ad)
        assert_array_almost_equal(B_DISC_BL_2,bd)
        assert_array_almost_equal(C_DISC_BL_2,cd)
        assert_array_almost_equal(D_DISC_BL_2,dd)

        ad,bd,cd,dd = ss_c2d(A_CONT, B_CONT, C_CONT, D_CONT, DT_DISC_3,
                             method='bilinear')

        assert_array_almost_equal(A_DISC_BL_3,ad)
        assert_array_almost_equal(B_DISC_BL_3,bd)
        assert_array_almost_equal(C_DISC_BL_3,cd)
        assert_array_almost_equal(D_DISC_BL_3,dd)

    def test_transferfunction(self):
        num,den = tf_c2d(NUM_CONT, DEN_CONT, DT_DISC, method='zoh')

        assert_array_almost_equal(NUM_DISC,num)
        assert_array_almost_equal(DEN_DISC,den)


if __name__ == "__main__":
    run_module_suite()
