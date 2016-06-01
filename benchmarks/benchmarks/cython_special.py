# This file is automatically generated by generate_ufuncs.py.
# Do not edit manually!

from __future__ import division, absolute_import, print_function

import numpy as np
from scipy import special

try:
    from scipy.special import cython_special
except ImportError:
    pass

from .common import Benchmark, with_attributes


class CythonSpecial(Benchmark):
    params = [(10, 100, 1000), ('python', 'numpy', 'cython')]
    param_names = ['argument', 'N', 'api']

    def setup(self, args, N, api):
        self.obj = []
        for arg in args:
            self.obj.append(arg*np.ones(N))
        self.obj = tuple(self.obj)

    @with_attributes(params=[((0.25, 0.75),)] + params, param_names=param_names)
    def time_beta_dd(self, args, N, api):
        if api == 'python':
            cython_special._bench_beta_dd_py(N, *args)
        elif api == 'numpy':
            special.beta(*self.obj)
        else:
            cython_special._bench_beta_dd_cy(N, *args)

    @with_attributes(params=[((1,),)] + params, param_names=param_names)
    def time_erf_d(self, args, N, api):
        if api == 'python':
            cython_special._bench_erf_d_py(N, *args)
        elif api == 'numpy':
            special.erf(*self.obj)
        else:
            cython_special._bench_erf_d_cy(N, *args)

    @with_attributes(params=[(((1+1j),),)] + params, param_names=param_names)
    def time_erf_D(self, args, N, api):
        if api == 'python':
            cython_special._bench_erf_D_py(N, *args)
        elif api == 'numpy':
            special.erf(*self.obj)
        else:
            cython_special._bench_erf_D_cy(N, *args)

    @with_attributes(params=[((1e-06,),)] + params, param_names=param_names)
    def time_exprel_d(self, args, N, api):
        if api == 'python':
            cython_special._bench_exprel_d_py(N, *args)
        elif api == 'numpy':
            special.exprel(*self.obj)
        else:
            cython_special._bench_exprel_d_cy(N, *args)

    @with_attributes(params=[((100,),)] + params, param_names=param_names)
    def time_gamma_d(self, args, N, api):
        if api == 'python':
            cython_special._bench_gamma_d_py(N, *args)
        elif api == 'numpy':
            special.gamma(*self.obj)
        else:
            cython_special._bench_gamma_d_cy(N, *args)

    @with_attributes(params=[(((100+100j),),)] + params, param_names=param_names)
    def time_gamma_D(self, args, N, api):
        if api == 'python':
            cython_special._bench_gamma_D_py(N, *args)
        elif api == 'numpy':
            special.gamma(*self.obj)
        else:
            cython_special._bench_gamma_D_cy(N, *args)

    @with_attributes(params=[((1, 1),)] + params, param_names=param_names)
    def time_jv_dd(self, args, N, api):
        if api == 'python':
            cython_special._bench_jv_dd_py(N, *args)
        elif api == 'numpy':
            special.jv(*self.obj)
        else:
            cython_special._bench_jv_dd_cy(N, *args)

    @with_attributes(params=[((1, (1+1j)),)] + params, param_names=param_names)
    def time_jv_dD(self, args, N, api):
        if api == 'python':
            cython_special._bench_jv_dD_py(N, *args)
        elif api == 'numpy':
            special.jv(*self.obj)
        else:
            cython_special._bench_jv_dD_cy(N, *args)

    @with_attributes(params=[((0.5,),)] + params, param_names=param_names)
    def time_logit_d(self, args, N, api):
        if api == 'python':
            cython_special._bench_logit_d_py(N, *args)
        elif api == 'numpy':
            special.logit(*self.obj)
        else:
            cython_special._bench_logit_d_cy(N, *args)

    @with_attributes(params=[((1,),)] + params, param_names=param_names)
    def time_psi_d(self, args, N, api):
        if api == 'python':
            cython_special._bench_psi_d_py(N, *args)
        elif api == 'numpy':
            special.psi(*self.obj)
        else:
            cython_special._bench_psi_d_cy(N, *args)

    @with_attributes(params=[((1,),)] + params, param_names=param_names)
    def time_psi_D(self, args, N, api):
        if api == 'python':
            cython_special._bench_psi_D_py(N, *args)
        elif api == 'numpy':
            special.psi(*self.obj)
        else:
            cython_special._bench_psi_D_cy(N, *args)
