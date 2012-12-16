import numpy as np

from _odebase import OdeSolverBase, NonReEntrantOdeSolverBase, OdeStepResult
import _lsoda

class LsodaSolver(NonReEntrantOdeSolverBase):

    messages = {
        2: "Integration successful.",
        -1: "Excess work done on this call (perhaps wrong Dfun type).",
        -2: "Excess accuracy requested (tolerances too small).",
        -3: "Illegal input detected (internal error).",
        -4: "Repeated error test failures (internal error).",
        -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
        -6: "Error weight became zero during problem.",
        -7: "Internal workspace insufficient to finish (internal error)."
    }

    def __init__(self,
                 rfn,
                 jacfn=None,
                 rtol=1e-6, atol=1e-12,
                 lband=None, uband=None,
                 nsteps=500,
                 max_step=0.0,  # corresponds to infinite
                 min_step=0.0,
                 first_step=0.0,  # determined by solver
                 ixpr=0,
                 max_hnil=0,
                 max_order_ns=12,
                 max_order_s=5,
                 method=None,
                 f_params=None,
                 jac_params=None
                 ):

        self.rfn = rfn

        self.jacfn = jacfn
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband

        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.ixpr = ixpr
        self.max_hnil = max_hnil

        self.f_params = None
        self.jac_params = None

    def _reset(self, n):
        # Calculate parameters for Fortran subroutine dvode.
        if self.jacfn is not None:
            if self.mu is None and self.ml is None:
                jt = 1
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 4
        else:
            if self.mu is None and self.ml is None:
                jt = 2
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 5
        lrn = 20 + (self.max_order_ns + 4) * n
        if jt in [1, 2]:
            lrs = 22 + (self.max_order_s + 4) * n + n * n
        elif jt in [4, 5]:
            lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
        else:
            raise ValueError('Unexpected jt=%s' % jt)
        lrw = max(lrn, lrs)
        liw = 20 + n
        rwork = np.zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = np.zeros((liw,), np.intc)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.ixpr
        iwork[5] = self.nsteps
        iwork[6] = self.max_hnil
        iwork[7] = self.max_order_ns
        iwork[8] = self.max_order_s
        self.iwork = iwork
        self.args = [self.rfn, None, None, None,
                     self.rtol, self.atol, 1, 1,
                     self.rwork, self.iwork,
                     self.jacobian, jt,
                     self.f_params, 0,
                     self.jac_params]

    def init_step(self, t0, y0):
        self.acquire_new_handle()

        y0 = np.asarray(y0)
        self._reset(y0.size)
        self.args[1] = y0.copy()
        self.args[2] = float(t0)
        self.args[3] = float(t0)

    def _step(self, t, y_out, itask):
        self.check_handle()

        self.args[3] = t
        self.args[6] = itask

        t, istate = _lsoda.lsoda(*self.args)
        y_out[...] = self.args[1]
        if istate < 0:
            return OdeStepResult(success=False, t=t)
        else:
            # upgrade istate
            self.args[7] = 2
        return OdeStepResult(success=True, t=t)

    def step(self, t_max, y_out):
        return self._step(t_max, y_out, itask=1)

    _itasks = {
        (False, False): 1,
        (True, False): 5,
        (False, True): 3,
        (True, True): 2
    }

    def step_auto(self, t_max, y_out, before=True, after=False):
        self.rwork[0] = t_max
        return self._step(t_max, y_out,
                          itask=self._itasks(bool(before), bool(after)))
