#!/bin/env python
from __future__ import division, absolute_import, print_function

import sys
import os
import re
import platform
import warnings

import numpy.distutils.system_info

from numpy.distutils.system_info import (
    system_info,
    mkl_info, lapack_mkl_info, blas_mkl_info, atlas_info, atlas_blas_info,
    lapack_atlas_info, atlas_threads_info, atlas_blas_threads_info,
    lapack_atlas_threads_info,
    blas_info, lapack_opt_info, blas_opt_info,
    get_platform, get_info, dict_append,
    AtlasNotFoundError, LapackNotFoundError, LapackSrcNotFoundError,
    BlasNotFoundError, BlasSrcNotFoundError
    )

def monkeypatch_numpy_distutils():
    if not hasattr(numpy.distutils, 'check_blas_fortran_abi'):
        numpy.distutils.system_info.mkl_info = hacked_mkl_info
        numpy.distutils.system_info.blas_mkl_info = hacked_blas_mkl_info
        numpy.distutils.system_info.lapack_mkl_info = hacked_lapack_mkl_info
        numpy.distutils.system_info.atlas_info = hacked_atlas_info
        numpy.distutils.system_info.atlas_blas_info = hacked_atlas_blas_info
        numpy.distutils.system_info.lapack_atlas_info = hacked_lapack_atlas_info
        numpy.distutils.system_info.blas_info = hacked_blas_info
        numpy.distutils.system_info.blas_opt_info = hacked_blas_opt_info
        numpy.distutils.system_info.lapack_opt_info = hacked_lapack_opt_info

def _set_info(self, **info):
    if info:
        if not check_blas_fortran_abi(self.blas_name, info):
            return
    return system_info.set_info(self, **info)

class hacked_mkl_info(mkl_info):
    blas_name = "MKL"
    set_info = _set_info

class hacked_lapack_mkl_info(lapack_mkl_info):
    blas_name = "MKL"
    set_info = _set_info

class hacked_blas_mkl_info(blas_mkl_info):
    blas_name = "MKL"
    set_info = _set_info

class hacked_atlas_info(atlas_info):
    blas_name = "ATLAS"
    set_info = _set_info

class hacked_atlas_blas_info(atlas_blas_info):
    blas_name = "ATLAS"
    set_info = _set_info

class hacked_lapack_atlas_info(lapack_atlas_info):
    blas_name = "ATLAS"
    set_info = _set_info

class hacked_atlas_threads_info(atlas_threads_info):
    blas_name = "PTATLAS"
    set_info = _set_info

class hacked_atlas_blas_threads_info(atlas_blas_threads_info):
    blas_name = "PTATLAS"
    set_info = _set_info

class hacked_lapack_atlas_threads_info(lapack_atlas_threads_info):
    blas_name = "PTATLAS"
    set_info = _set_info

class hacked_blas_info(blas_info):
    blas_name = "Generic BLAS"
    set_info = _set_info

class hacked_lapack_opt_info(lapack_opt_info):
    blas_name = "LAPACK"
    set_info = _set_info

    def calc_info(self):
        if sys.platform == 'darwin' and not os.environ.get('ATLAS', None):
            args = []
            link_args = []
            if get_platform()[-4:] == 'i386' or 'intel' in get_platform() or \
               'i386' in platform.platform():
                intel = 1
            else:
                intel = 0
            if os.path.exists('/System/Library/Frameworks'
                              '/Accelerate.framework/'):
                if intel:
                    args.extend(['-msse3'])
                else:
                    args.extend(['-faltivec'])
                link_args.extend(['-Wl,-framework', '-Wl,Accelerate'])
            elif os.path.exists('/System/Library/Frameworks'
                                '/vecLib.framework/'):
                if intel:
                    args.extend(['-msse3'])
                else:
                    args.extend(['-faltivec'])
                link_args.extend(['-Wl,-framework', '-Wl,vecLib'])
            if args:
                info = dict(extra_compile_args=args,
                            extra_link_args=link_args,
                            define_macros=[('NO_ATLAS_INFO', 3)])
                if check_blas_fortran_abi("Accelerate/Veclib", info):
                    self.set_info(**info)
                    return

        lapack_mkl_info = get_info('lapack_mkl')
        if lapack_mkl_info:
            self.set_info(**lapack_mkl_info)
            return

        atlas_info = get_info('atlas_threads')
        if not atlas_info:
            atlas_info = get_info('atlas')
        #atlas_info = {} ## uncomment for testing
        need_lapack = 0
        need_blas = 0
        info = {}
        if atlas_info:
            l = atlas_info.get('define_macros', [])
            if ('ATLAS_WITH_LAPACK_ATLAS', None) in l \
                   or ('ATLAS_WITHOUT_LAPACK', None) in l:
                need_lapack = 1
            info = atlas_info

        else:
            warnings.warn(AtlasNotFoundError.__doc__)
            need_blas = 1
            need_lapack = 1
            dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])

        if need_lapack:
            lapack_info = get_info('lapack')
            #lapack_info = {} ## uncomment for testing
            if lapack_info:
                dict_append(info, **lapack_info)
            else:
                warnings.warn(LapackNotFoundError.__doc__)
                lapack_src_info = get_info('lapack_src')
                if not lapack_src_info:
                    warnings.warn(LapackSrcNotFoundError.__doc__)
                    return
                dict_append(info, libraries=[('flapack_src', lapack_src_info)])

        if need_blas:
            blas_info = get_info('blas')
            #blas_info = {} ## uncomment for testing
            if blas_info:
                dict_append(info, **blas_info)
            else:
                warnings.warn(BlasNotFoundError.__doc__)
                blas_src_info = get_info('blas_src')
                if not blas_src_info:
                    warnings.warn(BlasSrcNotFoundError.__doc__)
                    return
                dict_append(info, libraries=[('fblas_src', blas_src_info)])

        self.set_info(**info)
        return


class hacked_blas_opt_info(blas_opt_info):
    blas_name = "BLAS"
    set_info = _set_info

    def calc_info(self):

        if sys.platform == 'darwin' and not os.environ.get('ATLAS', None):
            args = []
            link_args = []
            if get_platform()[-4:] == 'i386' or 'intel' in get_platform() or \
               'i386' in platform.platform():
                intel = 1
            else:
                intel = 0
            if os.path.exists('/System/Library/Frameworks'
                              '/Accelerate.framework/'):
                if intel:
                    args.extend(['-msse3'])
                else:
                    args.extend(['-faltivec'])
                args.extend([
                    '-I/System/Library/Frameworks/vecLib.framework/Headers'])
                link_args.extend(['-Wl,-framework', '-Wl,Accelerate'])
            elif os.path.exists('/System/Library/Frameworks'
                                '/vecLib.framework/'):
                if intel:
                    args.extend(['-msse3'])
                else:
                    args.extend(['-faltivec'])
                args.extend([
                    '-I/System/Library/Frameworks/vecLib.framework/Headers'])
                link_args.extend(['-Wl,-framework', '-Wl,vecLib'])
            if args:
                info = dict(extra_compile_args=args,
                            extra_link_args=link_args,
                            define_macros=[('NO_ATLAS_INFO', 3)])
                if check_blas_fortran_abi("Accelerate/Veclib", info):
                    self.set_info(**info)
                    return

        blas_mkl_info = get_info('blas_mkl')
        if blas_mkl_info:
            self.set_info(**blas_mkl_info)
            return

        atlas_info = get_info('atlas_blas_threads')
        if not atlas_info:
            atlas_info = get_info('atlas_blas')
        need_blas = 0
        info = {}
        if atlas_info:
            info = atlas_info
        else:
            warnings.warn(AtlasNotFoundError.__doc__)
            need_blas = 1
            dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])

        if need_blas:
            blas_info = get_info('blas')
            if blas_info:
                dict_append(info, **blas_info)
            else:
                warnings.warn(BlasNotFoundError.__doc__)
                blas_src_info = get_info('blas_src')
                if not blas_src_info:
                    warnings.warn(BlasSrcNotFoundError.__doc__)
                    return
                dict_append(info, libraries=[('fblas_src', blas_src_info)])

        self.set_info(**info)
        return



def check_blas_fortran_abi(blas_name, info):
    """
    Compile and run a test program to check whether the given BLAS
    conforms to the ABI of the Fortran compiler.

    The ABI is checked for the main suspect functions: SDOT, DDOT,
    CDOTU, ZDOTU.

    """

    from numpy.distutils.core import get_distribution
    from numpy.distutils.command.config import config as Config
    from distutils.ccompiler import CompileError, LinkError

    if not info:
        return False

    dist = get_distribution(True)
    config = Config(dist)
    options = dist.command_options.get('config')
    if options:
        dist._set_command_options('config', config, options)

    body = """\
      program main
      external sdot, ddot, cdotu, zdotu
      real sx(1), sy(1), sa, sdot
      double precision dx(1), dy(1), da, ddot
      complex cx(1), cy(1), ca, cdotu
      double complex zx(1), zy(1), za, zdotu

      sx(1) = 1e0
      sy(1) = 2e0
      sa = sdot(1, sx, 1, sy, 1)
      if (sa.ne.sx(1)*sy(1)) stop 1

      dx(1) = 1d0
      dy(1) = 2d0
      da = ddot(1, dx, 1, dy, 1)
      if (da.ne.dx(1)*dy(1)) stop 2

      cx(1) = (1e0, 2e0)
      cy(1) = (3e0, 4e0)
      ca = cdotu(1, cx, 1, cy, 1)
      if (ca.ne.cx(1)*cy(1)) stop 3

      zx(1) = (1d0, 2d0)
      zy(1) = (3d0, 4d0)
      za = zdotu(1, zx, 1, zy, 1)
      if (za.ne.zx(1)*zy(1)) stop 4

      write(*,*) 'BLAS', 'SUCCESS'
      end
    """

    libraries = info.get('libraries', [])
    library_dirs = info.get('library_dirs', [])
    extra_compile_args = info.get('extra_compile_args', [])
    extra_link_args = info.get('extra_link_args', [])

    # The distutils config API does not offer a way to pass
    # extra_*_args to the compiler. Therefore, we monkeypatch the
    # active compiler to inject the arguments. (The Fortran compiler
    # classes originate from numpy.distutils so that we are not
    # monkeypatching another library.)

    def new_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        if extra_postargs:
            extra_postargs += extra_compile_args
        else:
            extra_postargs = extra_compile_args
        return old_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)

    def new_link(self, target_desc, objects,
                 output_filename, output_dir=None, libraries=None,
                 library_dirs=None, runtime_library_dirs=None,
                 export_symbols=None, debug=0, extra_preargs=None,
                 extra_postargs=None, build_temp=None, target_lang=None):
        if extra_postargs:
            extra_postargs += extra_link_args
        else:
            extra_postargs = extra_link_args
        return old_link(self, target_desc, objects,
                        output_filename, output_dir, libraries,
                        library_dirs, runtime_library_dirs,
                        export_symbols, debug, extra_preargs,
                        extra_postargs, build_temp, target_lang)

    config._check_compiler()

    if config.fcompiler is None:
        # No Fortran compiler, so checking the ABI is not needed.
        return True

    old_compile = config.fcompiler.__class__._compile
    old_link = config.fcompiler.__class__.link
    try:
        config.fcompiler.__class__._compile = new_compile
        config.fcompiler.__class__.link = new_link

        # Run the test program
        exitcode, output = config.get_output(body,
                                             libraries=libraries,
                                             library_dirs=library_dirs,
                                             lang="f77")
    finally:
        config.fcompiler.__class__._compile = old_compile
        config.fcompiler.__class__.link = old_link

    # Note: get_output includes also `body` in the output, so be careful
    # in checking the success status. Also, Fortran program exit codes 
    # are undefined.
    is_abi_compatible = output and re.search(r'BLAS\s*SUCCESS', output, re.S)

    if not is_abi_compatible:
        import textwrap
        msg = textwrap.dedent("""

        ***********************************************************************
        WARNING:

        BLAS library (%s) detected, but its
        Fortran ABI is incompatible with the selected Fortran compiler.
        It is therefore not used now.

        If you are using GNU Fortran Compiler on OSX, setting
        the environment variable FFLAGS=\"-arch i386 -arch x86_64 -fPIC -ff2c\"
        may fix this issue.
        ***********************************************************************

        """
        % (blas_name,))
        warnings.warn(msg)

    return is_abi_compatible
