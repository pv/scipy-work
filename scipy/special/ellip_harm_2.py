from . import _ellip_harm_2
from _ellip_harm_2 import _ellipsoid
import threading
import numpy as np

_ellip_lock = threading.Lock()

def ellip_harm_2(h2, k2, n, p, s):
    return _ellipsoid(h2, k2, n, p, s)
