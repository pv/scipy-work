from __future__ import division, absolute_import, print_function

import sys
import os
import re
import warnings
import platform

from numpy.distutils.system_info import (
    get_info as _get_info,
    system_info, dict_append, get_platform,
    BlasNotFoundError, LapackNotFoundError,
    AtlasNotFoundError, LapackSrcNotFoundError,
    BlasSrcNotFoundError
    )

def get_info(name, notfound_action=0):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """

    cl = {
          'blas': blas_info,                  # use blas_opt instead
          'lapack': lapack_info,              # use lapack_opt instead
          'lapack_opt': lapack_opt_info,
          'blas_opt': blas_opt_info,
          }.get(name.lower(), None)

    if cl is None:
        return _get_info(name, notfound_action)
    else:
        return cl().get_info(notfound_action)


class lapack_info(system_info):
    section = 'lapack'
    dir_env_var = 'LAPACK'
    _lib_names = ['lapack']
    notfounderror = LapackNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        lapack_libs = self.get_libs('lapack_libs', self._lib_names)
        info = self.check_libs(lib_dirs, lapack_libs, [])
        if info is None:
            return
        info['language'] = 'f77'

        if not check_lapack_fortran_abi("Generic LAPACK", info):
            return

        self.set_info(**info)


class lapack_opt_info(system_info):

    notfounderror = LapackNotFoundError

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
                if check_lapack_fortran_abi("Accelerate/Veclib", info):
                    self.set_info(**info)
                    return

        lapack_mkl_info = get_info('lapack_mkl')
        if lapack_mkl_info and check_lapack_fortran_abi("Intel MKL", info):
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

        if not check_lapack_fortran_abi("Atlas", info):
            return

        self.set_info(**info)
        return


class blas_opt_info(system_info):
    notfounderror = BlasNotFoundError

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
                if check_blas_fortran_abi("Accelerate", info):
                    self.set_info(**info)
                    return

        blas_mkl_info = get_info('blas_mkl')
        if blas_mkl_info and check_blas_fortran_abi("MKL", blas_mkl_info):
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

        if not check_blas_fortran_abi("ATLAS", info):
            return

        self.set_info(**info)

class blas_info(system_info):
    section = 'blas'
    dir_env_var = 'BLAS'
    _lib_names = ['blas']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        blas_libs = self.get_libs('blas_libs', self._lib_names)
        info = self.check_libs(lib_dirs, blas_libs, [])
        if info is None:
            return

        if not check_blas_fortran_abi("Generic BLAS", info):
            return

        info['language'] = 'f77'  # XXX: is it generally true?
        self.set_info(**info)


def check_blas_fortran_abi(blas_name, info):
    """
    Compile and run a test program to check whether the given BLAS
    conforms to the ABI of the Fortran compiler.

    The ABI is checked for the main suspect functions: SDOT, DDOT,
    CDOTU, ZDOTU.

    """
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

      write(*,*) 'EXIT', 'SUCCESS'
      end
    """
    return _check_fortran_abi(blas_name, info, body)


def check_lapack_fortran_abi(lapack_name, info):
    """
    Compile and run a test program to check whether the given LAPACK
    conforms to the ABI of the Fortran compiler.

    The ABI is checked for the main suspect functions:
    CLADIV, ZLADIV

    """
    body = """\
      program main
      external cladiv, zladiv
      complex cx, cy, cz, cladiv
      double complex zx, zy, zz, zladiv

      cx = (1e0, 0e0)
      cy = (1e0, 0e0)
      cz = cladiv(cx, cy)
      if (cz.ne.cx/cy) stop 1

      zx = (1e0, 0e0)
      zy = (1e0, 0e0)
      zz = zladiv(zx, zy)
      if (zz.ne.zx/zy) stop 1

      write(*,*) 'EXIT', 'SUCCESS'
      end
    """
    return _check_fortran_abi(lapack_name, info, body)


def _check_fortran_abi(lib_name, info, body):
    """
    Compile and run a test program to check whether it succeeds.

    """

    from numpy.distutils.core import get_distribution
    from numpy.distutils.command.config import config as Config

    if not info:
        return False

    dist = get_distribution(True)
    config = Config(dist)
    options = dist.command_options.get('config')
    if options:
        dist._set_command_options('config', config, options)

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
    is_abi_compatible = output and re.search(r'EXIT\s*SUCCESS', output, re.S)

    if not is_abi_compatible:
        import textwrap
        msg = textwrap.dedent("""

        *******************************************************************
        WARNING:

        Library (%s) detected, but its
        Fortran ABI is incompatible with the selected Fortran compiler.
        It is therefore not used now.

        If you are using GNU Fortran Compiler, setting the environment
        variable FOPT=\"-O2 -ff2c\" may fix this issue
        *******************************************************************

        """
        % (lib_name,))
        warnings.warn(msg)

    return is_abi_compatible
