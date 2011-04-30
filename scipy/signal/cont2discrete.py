"""
cont2discrete - Continuous to Discrete state-space transforms
"""

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# March 29, 2011

import numpy
import numpy.linalg
import scipy.linalg
from math import sqrt
import math

from ltisys import tf2ss, ss2tf

def _mrdivide(b,a):
    """Convenience function for matrix divides"""
    s = numpy.linalg.solve(a.transpose(),b.transpose())
    return s.transpose()

def cont2discrete(*args,**kwargs):
    """Transforms a continuous state-space system to a discrete state-space
    system.  The function defaults to a bilinear transform.
    
    Parameters
    -----------
    a,b,c,d : ndarray
        Arrays representing the continuous state-space system
        
    dt : float
        The discretization time step
        
    method : string
        Which method to use (bilinear or zoh).  Defaults to zoh.
        
    Returns
    -------
    ad,bd,cd,dd : ndarray
        The equivalent discrete state-space system
        
    Method
    ------
    By default, the routine uses a Zero-Order Hold (zoh) method
    to perform the transformation.  Alternatively, Tustin's 
    bilinear approximation can be used.
    """
    
    
    
    # 3 args = transfer function
    if len(args) == 3:
        a,b,c,d = tf2ss(args[0],args[1])
        dt = args[2]
    # 5 args = state-space system
    elif len(args) == 5:
        a,b,c,d,dt = args
    else:
        raise ValueError("Function accepts 3 (tf) or 5 (ss) arguments")
    
    try:
        method = kwargs['method']
    except KeyError:
        # Default method is zero-order hold
        method='zoh'
    
    if method=='bilinear' or method=='tustin':
    
        itv = 2.0/dt*numpy.eye(a.shape[0])
    
        # ad = (itv+a)/(itv-a)
        ad = _mrdivide((itv+a),(itv-a))
        iab = numpy.linalg.solve((itv-a),b)
    
        tk = 2.0/dt #sqrt(2.0/dt*dt)
        bd = tk*iab
    
        cd = 2.0*_mrdivide(c,(itv-a))

        dd = d + numpy.asmatrix(c)*numpy.asmatrix(iab)

    elif method=='zoh':
        
        em = numpy.vstack(( numpy.hstack(( a, b )), \
                            numpy.hstack(( numpy.zeros((b.shape[1],a.shape[1])), numpy.zeros((b.shape[1],b.shape[1])) )) ))
        
        ms = scipy.linalg.expm(dt*em)
        
        ms = ms[0:a.shape[0]]
        ad = ms[:,0:a.shape[1]]
        bd = ms[:,a.shape[1]:]
        
        cd = c
        dd = d
    
    else:
        
        raise ValueError("Unknown transformation method.")
    
    if len(args) == 3:
        return ss2tf(ad,bd,cd,dd)
    else:
        return ad,bd,cd,dd
    