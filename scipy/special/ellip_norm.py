from . import ellip_normal
from ellip_normal import _ellipsoid_norm
import threading
import numpy as np

_ellip_lock = threading.Lock()

def ellip_normal(h2, k2, n, p):
    return _ellipsoid_norm(h2, k2, n, p)
