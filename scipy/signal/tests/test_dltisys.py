
# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# April 4, 2011

import numpy
import numpy.testing 
from scipy.signal import dlsim, dstep, dimpulse

DLSIM_A = numpy.asarray([[0.9,0.1],[-0.2,0.9]])
DLSIM_B = numpy.asarray([[0.4,0.1,-0.1],[0.0,0.05,0.0]])
DLSIM_C = numpy.asarray([[0.1,0.3],])
DLSIM_D = numpy.asarray([[0.0,-0.1,0.0],])
DLSIM_DT = 0.5

DLSIM_U = numpy.hstack( ( numpy.asmatrix(numpy.linspace(0,4.0,num=5)).transpose(),0.01*numpy.ones((5,1)),-0.002*numpy.ones((5,1)) ) )

DLSIM_YOUT = numpy.asmatrix([-0.001,-0.00073,0.039446,0.0915387,0.13195948]).transpose()
DLSIM_XOUT = numpy.asarray([[0,0],[0.0012,0.0005],[0.40233,0.00071],[1.163368,-0.079327,],[2.2402985,-0.3035679]])

DLSIM_NUM = numpy.asarray([1.0,-0.1])
DLSIM_DEN = numpy.asarray([0.3,1.0,0.2])
DLSIM_TFYOUT = numpy.asmatrix([0.0,0.0,3.33333333333333,-4.77777777777778,23.0370370370370]).transpose()

DLSIM_STEPOUT = (numpy.asarray([0.0,0.04,0.052,0.0404,0.00956,-0.036324,-0.093318,-0.15782348,-0.226628324,-0.2969374948]),
                 numpy.asarray([-0.1,-0.075,-0.058,-0.04815,-0.04453,-0.0461895,-0.0521812,-0.061588875,-0.073549579,-0.08727047595]),
                 numpy.asarray([0.0,-0.01,-0.013,-0.0101,-0.00239,0.009081,0.0233295,0.03945587,0.056657081,0.0742343737]) )
                 
DLSIM_IMPULSEOUT = (numpy.asarray([0.0,0.04,0.012,-0.0116,-0.03084,-0.045884,-0.056994,-0.06450548,-0.068804844,-0.0703091708]),
                    numpy.asarray([-0.1,0.025,0.017,0.00985,0.00362,-0.0016595,-0.0059917,-0.009407675,-0.011960704,-0.01372089695]),
                    numpy.asarray([0.0,-0.01,-0.003,0.0029,0.00771,0.011471,0.0142485,0.01612637,0.017201211,0.0175772927]))

class TestDLTI(numpy.testing.TestCase):
    
    def _test_dlsim(self):
    
        t_in = numpy.linspace(0,2.0,num=5)
        tout,yout,xout = dlsim((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT), DLSIM_U, t_in) 

        numpy.testing.assert_array_almost_equal(yout,DLSIM_YOUT)
        numpy.testing.assert_array_almost_equal(xout,DLSIM_XOUT)
        numpy.testing.assert_array_almost_equal(t_in,tout)
        
        # Interpolated control
        u_sparse = DLSIM_U[[0,4],:]
        t_sparse = numpy.asarray([0.0,2.0])
        
        tout,yout,xout = dlsim((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT), u_sparse, t_sparse) 

        numpy.testing.assert_array_almost_equal(yout,DLSIM_YOUT)
        numpy.testing.assert_array_almost_equal(xout,DLSIM_XOUT)
        numpy.testing.assert_equal(len(tout),yout.shape[0])
        
        # Transfer functions
        tout,yout = dlsim((DLSIM_NUM,DLSIM_DEN,DLSIM_DT), DLSIM_U[:,0], t_in)
        numpy.testing.assert_array_almost_equal(yout,DLSIM_TFYOUT)
        numpy.testing.assert_array_almost_equal(t_in,tout)

    def test_dstep(self):
        tout,yout = dstep((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT),n=10)
        
        numpy.testing.assert_equal(len(yout),3)
        
        for i in range(0,len(yout)):
            numpy.testing.assert_equal(yout[i].shape[0],10)
            numpy.testing.assert_array_almost_equal(yout[i].flatten(),DLSIM_STEPOUT[i])
            
    def test_dimpulse(self):
        tout,yout = dimpulse((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT),n=10)
        
        numpy.testing.assert_equal(len(yout),3)
        numpy.testing.assert_equal(yout[0].shape[0],10)
        numpy.testing.assert_equal(yout[1].shape[0],10)
        
        for i in range(0,len(yout)):
            numpy.testing.assert_equal(yout[i].shape[0],10)
            numpy.testing.assert_array_almost_equal(yout[i].flatten(),DLSIM_IMPULSEOUT[i])
            