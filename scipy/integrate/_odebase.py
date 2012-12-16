from collections import namedtuple
import numpy as np

__all__ = ['OdeSolverBase', 'OdeConcurrencyError', 'NonReEntrantOdeSolverBase']

OdeSolveResult = namedtuple("OdeSolveResult",
                            ['success', 't', 'y', 't_err', 'y_err'])

OdeStepResult = namedtuple("OdeStepResult",
                           ['success', 't'])

class OdeSolverBase(object):
    """
    The interface which ODE solvers must implement.
    """

    dtype = np.dtype(float)

    def __init__(self, rfn, **options):
        """ 
        Initialize the ODE Solver and it's default values 

        Parameters
        ----------
        rfn
            Right-hand-side function, with signature::

                def rfn(t, y, out):
                    out[...] = something

        options
            Additional options for initialization.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def solve(self, tspan, y0):
        """
        Runs the solver.
        
        Parameters
        ----------
        tspan : array-like
            Times at which the computed value will be returned.
            Must contain the start time.
        y0 : array-like
            Initial values.

        Returns
        -------
        result : OdeSolveResult
            A named tuple with key names:
            
            success
                indicating return status of the solver
            t
                numpy array of times at which the computations were successful
            y
                numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            t_err
                float or None - if recoverable error occured (for example reached maximum
                number of allowed iterations), this is the time at which it happened
            y_err
                array of values corresponding to time t_err, or None
        """
        # Default implementation: solvers should override this for better efficiency.

        tspan = np.asarray(tspan)
        y0 = np.asarray(y0)

        t = np.zeros(tspan.shape, dtype=float)
        y = np.zeros(tspan.shape + y0.shape, dtype=self.dtype)

        if len(tspan) > 0:
            self.init_step(tspan[0], y0)
            y[0] = y0

        for j in xrange(0, len(tspan)):
            r = self.step(t[j], y[j])
            if not r.success:
                return OdeSolveResult(success=False, t=t[:j], y=y[:j], t_err=None, y_err=None)

        return OdeSolveResult(success=True, t=t, y=y, t_err=None, y_err=None)

    def solve_auto(self, t_max, t0, y0, maxiter=100000):
        """
        Runs the solver, determining the time points automatically.

        Parameters
        ----------
        t_max : float
            Final time.
        t0 : float
            Initial time.
        y0 : array-like
            Initial values.
        maxiter : int, optional
            Maximum number of iterations to perform.

        Returns
        -------
        result : OdeSolveResult
            A named tuple with key names:
            
            success
                indicating return status of the solver
            t
                numpy array of times at which the computations were successful
            y
                numpy array of values corresponding to times t (values of y[i, :] ~ t[i])
            t_err
                float or None - if recoverable error occured (for example reached maximum
                number of allowed iterations), this is the time at which it happened
            y_err
                array of values corresponding to time t_err, or None
        """
        # Default implementation: solvers should override this for better efficiency.

        t_max = float(t_max)
        y0 = np.asarray(y0)

        nt = 0
        t = np.zeros((100,), dtype=float)
        y = np.zeros((100,) + y0.shape, dtype=self.dtype)

        self.init_step(t0, y0)
        t[0] = t0
        y[0] = y0
        nt += 1

        while True:
            for j in range(nt, min(maxiter, len(t))):
                r = self.step_auto(t_max, y[j], before=True, after=True)
                if not r.success:
                    return OdeSolveResult(success=False, t=t[:j], y=y[:j], t_err=None, y_err=None)
                elif r.t >= t_max:
                    t[j] = r.t
                    return OdeSolveResult(success=True, t=t[:j], y=y[:j], t_err=None, y_err=None)
                t[j] = r.t

            nt = j + 1

            if nt == maxiter:
                return OdeSolveResult(success=False, t=t[:nt], y=y[:nt],
                                      t_err=t[j], y_err=y[j])

            t.resize(2*nt + 1)
            y.resize(t.shape[0], y.shape[1])

    def init_step(self, t0, y0):
        """
        Initializes the solver and allocates memory.

        Parameters
        ----------
        t0 : array-like
            Initial time.
        y0 : array-like
            Initial condition for y.
        """
        raise NotImplementedError('all ODE solvers must implement this')

    def step(self, t, y_out):
        """
        Method for calling successive next step of the ODE solver to
        allow more precise control over the solver. The 'init_step'
        method has to be called before the 'step' method.

        Parameters
        ----------
        t : float
            If t > 0.0 then integration is performed until this time
            and results at this time are returned in y_retn.

            If t < 0.0 only one internal step is perfomed towards time abs(t)
            and results after this one time step are returned.
        y_out : array-like
            Array in which the computed value will be stored.
            Needs to be preallocated.

        Returns
        -------
        result : OdeStepResult
            A named tuple with keys

            success
                status of the computation (successful or error occured)
            t
                time, where the solver stopped (when no error occured, this
                is equal to the requested time)

        """
        raise NotImplementedError('all ODE solvers must implement this')

    def step_auto(self, t, y_out, before=True, after=False):
        """
        Take a step of length determined internally by the solver.

        Method for calling successive next step of the ODE solver to
        allow more precise control over the solver. The 'init_step'
        method has to be called before the 'step_auto' method.

        Parameters
        ----------
        t_max : float
            
        y_out : array-like
            Array in which the computed value will be stored.
            Needs to be preallocated.

        before : bool, optional
            Whether the solver is allowed to stop before time `t`.
            Only certain values may be supported by individual solvers.
        after : bool, optional
            Whether the solver is allowed to stop after time `t`
            Only certain values may be supported by individual solvers.

        Returns
        -------
        result : OdeStepResult
            A named tuple with keys

            success
                status of the computation (successful or error occured)
            t_out
                time, where the solver stopped (can differ from t even
                when successful)

        """
        raise NotImplementedError('This method is not implemented by this ODE solver')


class OdeConcurrencyError(RuntimeError):
    """
    Failure due to concurrent usage of an integrator that can be used
    only for a single problem at a time.

    """
    def __init__(self, name):
        msg = ("ODE solver `%s` can be used to solve only a single problem "
               "at a time. If you want to integrate multiple problems, "
               "consider using a different solver ") % name
        RuntimeError.__init__(self, msg)


class NonReEntrantOdeSolverBase(OdeSolverBase):
    """
    Some of the integrators have internal state (ancient Fortran...),
    and so only one instance can use them at a time.  This base class
    provides simple tools for dealing with this.

    """
    _active_reentrant_handle = 0

    def _acquire_new_handle(self):
        self.__class__._active_reentrant_handle = self.__class__._active_reentrant_handle + 1
        self._reentrant_handle = self.__class__._active_global_handle

    def _check_handle(self):
        if self._reentrant_handle is not self.__class__._active_global_handle:
            raise OdeConcurrencyError(self.__class__.__name__)
