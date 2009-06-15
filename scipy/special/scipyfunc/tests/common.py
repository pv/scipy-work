import csv

import numpy as np

def assert_tol_equal(a, b, rtol=1e-7, atol=0, err_msg='', verbose=True):
    """Assert that `a` and `b` are equal to tolerance ``atol + rtol*abs(b)``"""
    def compare(x, y):
        return np.allclose(x, y, rtol=rtol, atol=atol)
    a, b = np.asanyarray(a), np.asanyarray(b)
    header = 'Not equal to tolerance rtol=%g, atol=%g' % (rtol, atol)
    np.testing.utils.assert_array_compare(compare, a, b, err_msg=str(err_msg),
                                          verbose=verbose, header=header)

def parse_txt_data(filename):
    f = open(filename)
    try:
        reader = csv.reader(f, delimiter=',')
        data = []
        for row in reader:
            data.append(map(float, row))
        nc = len(data[0])
        for i in data:
            if not nc == len(i):
                raise ValueError(i)
        return np.array(data)
        ## guess number of columns/rows
        #row0 = f.readline()
        #nc = len(row0.split(',')) - 1
        #nlines = len(f.readlines()) + 1
        #f.seek(0)
        #data = np.fromfile(f, sep=',')
        #if not data.size == nc * nlines:
        #    raise ValueError("Inconsistency between array (%d items) and "
        #                     "guessed data size %dx%d" % (data.size, nlines, nc))
        #data = data.reshape((nlines, nc))
        #return data
    finally:
        f.close()

def nulps_array(a, b):
    t = np.common_type(a, b)
    if not t in [np.float32, np.float64]:
        raise ValueError("Could not convert both arrays to supported type")
    a = np.array(a, dtype=t)
    b = np.array(b, dtype=t)

    if not a.shape == b.shape:
        raise ValueError("a and b do not have the same shape")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Nan/Inf in array not supported yet")

    def _nulps(vdt, comp):
        r = np.zeros(a.shape, vdt)
        # Handle exact equality separately
        eid = (a == b)
        if eid.size > 1:
            r[a==b] = 0

        # Reinterpret float as integer, to compare the integer representation
        ra = a.view(vdt)
        rb = b.view(vdt)
        ra = comp - ra
        rb = comp - rb

        id = np.where(a != b)
        diff = np.array(ra-rb, dtype=vdt)
        r[id] = np.abs(diff)[id]
        return r

    if a.dtype == np.float32:
        return _nulps(np.int32, 0x80000000)
    elif a.dtype == np.float64:
        return _nulps(np.int64, 0x8000000000000000)
    else:
        raise ValueError("Unsupported dtype %s" % a.dtype)
