# Pytest customization
from __future__ import division, absolute_import, print_function

import os
import gc
import sys
import pytest
import warnings

from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning


def pytest_runtest_setup(item):
    mark = item.get_marker("xslow")
    if mark is not None:
        try:
            v = int(os.environ.get('SCIPY_XSLOW', '0'))
        except ValueError:
            v = False
        if not v:
            pytest.skip("very slow test; set environment variable SCIPY_XSLOW=1 to run it")


@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()

    if old_mode != new_mode:
        warnings.warn("FPU mode changed from {0:#x} to {1:#x} during "
                      "the test".format(old_mode, new_mode),
                      category=FPUModeChangeWarning, stacklevel=0)


if hasattr(sys, 'gettotalrefcount'):
    # For Python debug builds, check for reference leaks
    @pytest.hookimpl()
    def pytest_pyfunc_call(pyfuncitem):
        testfunction = pyfuncitem.obj
        funcargs = pyfuncitem.funcargs
        testargs = {}
        for arg in pyfuncitem._fixtureinfo.argnames:
            testargs[arg] = funcargs[arg]

        for k in range(2):
            old_count = None
            new_count = None

            gc.collect()
            old_count = sys.gettotalrefcount()
            testfunction(**testargs)
            gc.collect()
            new_count = sys.gettotalrefcount()

            if old_count == new_count:
                # No leak
                break

            # Re-run to see if refcount stabilises
            pyfuncitem.teardown()
            pyfuncitem.setup()
            gc.collect()
        else:
            raise AssertionError(
                "reference leak: sys.gettotalrefcount() changed "
                "by {0} during test".format(new_count - old_count))

        return True

