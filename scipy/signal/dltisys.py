"""
dltisys - Code related to discrete linear time-invariant systems
"""

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# April 4, 2011

import math
import numpy
from scipy.interpolate import interp1d
from scipy.signal import tf2ss

def dlsim(system, u, t=None, x0=None):
    """
    Simulate output of a discrete-time linear system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 3: (num, den, dt)
        * 5: (A, B, C, D, dt)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.
    X0 :
        The initial conditions on the state vector (zero by default).

    Returns
    -------
    tout : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time-evolution of the state-vector.  Only generated if the 
        input is a state-space systems.

    """

    if len(system) == 3:
        a,b,c,d = tf2ss(system[0],system[1])
        dt = system[2]
    elif len(system) == 5:
        a,b,c,d,dt = system
    else:
        raise ValueError("System argument should be a discrete transfer function or state-space system")
    
    if t is None:
        out_samples = max(u.shape)
        stoptime = (out_samples-1)*dt
    else:
        stoptime = t[-1]
        out_samples = int(math.floor(stoptime/dt))+1
    
    # Pre-build output arrays
    xout = numpy.zeros((out_samples,a.shape[0]))
    yout = numpy.zeros((out_samples,c.shape[0]))
    tout = numpy.linspace(0.0,stoptime,num=out_samples)
    
    # Check initial condition
    if x0 is None:
        xout[0,:] = numpy.zeros((a.shape[1],))
    else:
        xout[0,:] = x0
    
    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        u_dt_interp = interp1d(t,u.transpose(),copy=False,bounds_error=True)
        u_dt = u_dt_interp(tout)
        u_dt = u_dt.transpose()
    
    # Simulate the system
    for i in range(0,out_samples-1):
        
        xout[i+1,:] = numpy.dot(a,xout[i,:]) + numpy.dot(b,u_dt[i,:])
        yout[i,:] = numpy.dot(c,xout[i,:]) + numpy.dot(d,u_dt[i,:])
        
    # Last point
    yout[out_samples-1,:] = numpy.dot(c,xout[out_samples-1,:]) + numpy.dot(d,u_dt[out_samples-1,:])
    
    if len(system) == 3:
        return tout,yout
    elif len(system) == 5:
        return tout,yout,xout

def dimpulse(system, x0=None, t=None, n=None):
    """Impulse response of discrete-time system.

    Parameters
    ----------
    system : tuple
        If specified as a tuple, the system is described as
        ``(num, den)``, ``(zero, pole, gain)``, or ``(A, B, C, D)``.
    x0 : array_like, optional
        Initial state-vector.  Defaults to zero.
    T : array_like, optional
        Time points.  Computed if not given.
    N : int, optional
        The number of time points to compute (if `t` is not given).

    Returns
    -------
    t : ndarray
        A 1-D array of time points.
    yout : tuple of array_like
        Step response of system.  Each element of the tuple represents
        the output of the system based on an impulse in each input.

    """
    
    # Determine the system type and set number of inputs and time steps
    if len(system) == 3:
        n_inputs = 1
        dt = system[2]
    elif len(system) == 5:
        n_inputs = system[1].shape[1]
        dt = system[4]
    
    # Default to 100 samples if unspecified
    if n is None:
        n = 100
        
    # If time is not specified, use the number of samples
    # and system dt
    if t is None:
        t = numpy.arange(0,n*dt,dt)
        
    # For each input, implement a step change
    yout = None
    for i in range(0,n_inputs):
        
        u = numpy.zeros((t.shape[0],n_inputs))
        u[0,i] = 1.0

        one_output = dlsim(system, u, t=t, x0=x0)
        
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout+(one_output[1],)
            
        tout = one_output[0]
        
    return tout, yout
    
def dstep(system, x0=None, t=None, n=None):
    """Step response of discrete-time system.

    Parameters
    ----------
    system : a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation.
            3 (num, den, dt)
            5 (A, B, C, D, dt)
    x0 : array_like, optional
        Initial state-vector (default is zero).
    t : array_like, optional
        Time points (computed if not given).
    n : int
        Number of time points to compute if `t` is not given.

    Returns
    -------
    t : 1D ndarray
        Output time points.
    yout : tuple of array_like
        Step response of system.  Each element of the tuple represents
        the output of the system based on a step response to each input.

    """
    
    # Determine the system type and set number of inputs and time steps
    if len(system) == 3:
        n_inputs = 1
        dt = system[2]
    elif len(system) == 5:
        n_inputs = system[1].shape[1]
        dt = system[4]
    
    # Default to 100 samples if unspecified
    if n is None:
        n = 100
        
    # If time is not specified, use the number of samples
    # and system dt
    if t is None:
        t = numpy.arange(0,n*dt,dt)
        
    # For each input, implement a step change
    yout = None
    for i in range(0,n_inputs):
        
        u = numpy.zeros((t.shape[0],n_inputs))
        u[:,i] = numpy.ones((t.shape[0],))

        one_output = dlsim(system, u, t=t, x0=x0)
        
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout+(one_output[1],)
            
        tout = one_output[0]
        
    return tout, yout