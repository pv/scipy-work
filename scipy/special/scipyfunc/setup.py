#!/usr/bin/env python
import os
import sys
from os.path import join
from numpy.distutils.misc_util import get_npymath_info

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('scipyfunc', parent_package, top_path)

    # Extension _scipyfunc
    config.add_extension('_scipyfunc',
                         sources=[join('*.src'),
                                  join('*.c')],
                         depends=[join('*.h')],
                         f2py_options=['--no-wrap-functions'],
                         define_macros=[],
                         extra_info=get_npymath_info())
    config.add_include_dirs(join(os.path.dirname(__file__)))

    config.add_data_dir('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
