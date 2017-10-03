from __future__ import division, print_function, absolute_import

import os
from numpy.distutils.commands.build_ext import build_ext as old_build_ext


class build_ext(old_build_ext):
    def run(self):
        old_build_ext.run()

        if self.compiler.compiler_type == 'msvc':
            # Bundle the CRT
            runtime_lib_dir = os.path.join(
                self.build_lib, self.distribution.get_name(), 'extra-dll')

            if not os.path.isdir(runtime_lib_dir):
                pass

            import pdb; pdb.set_trace()
