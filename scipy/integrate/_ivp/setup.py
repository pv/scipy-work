#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

from os.path import join

from scipy._build_utils import numpy_nodepr_api


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('_ivp', parent_package, top_path)

    include_dirs = [join('..', '..', '_lib', 'src')]
    ccallback_src = [join(include_dirs[0], 'ccallback.h')]

    # lsoda
    config.add_extension('_lsoda',
                         sources=['_lsoda.c'],
                         include_dirs=include_dirs,
                         depends=ccallback_src)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
