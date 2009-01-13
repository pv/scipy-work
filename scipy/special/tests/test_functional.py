"""
Test functional relationships of special functions across a range of
parameters.

"""
import inspect
import numpy as np
import scipy.special as special

def _check_functional(func, spec=None, repeat=10, tol=1e-8):
    """
    Check that `func` evaluates to zero for various arguments.

    Parameters
    ----------
    func : f(*args)
        Function to check
    spec : str of length < len(args)
        Argument specification for function, one letter per item

        ===  =================================================================
        key  value
        ===  =================================================================
        C    complex value, may be exponentially big
        R    real value, may be exponentially big
        I    integer value, may be sys.maxint big
        N    positive integer, may be sys.maxint big
        c    complex value, of order unity
        r    real value, of order unity
        i    integer value, of order unity
        n    positive integer, of order unity
        ===  =================================================================

    """

    args, varargs, varkw, defaults = inspect.getargspec(func)
    if spec is None:
        spec = "C"*len(args)
    elif len(spec) < len(args):
        spec = spec + "C"*len(args)

    np.random.seed(1)
    rand = lambda: (np.random.rand() - 0.5)
    randint = lambda k: (np.random.randint(2*k) - k)
    
    def random_args(spec):
        args = []
        for k in spec:
            jj = 1j
            if k == 'C':
                if np.random.randint(5) == 0: jj = 1
                x = (rand() + jj*rand()) * 10**randint(10)
            elif k == 'c':
                if np.random.randint(5) == 0: jj = 1
                x = (rand() + jj*rand())
            elif k == 'R':
                x = rand() * 10**randint(10)
            elif k == 'r':
                x = rand() * 10**randint(10)
            elif k == 'I':
                x = randint(10**(1 + np.random.randint(8)))
            elif k == 'i':
                x = randint(10)
            elif k == 'N':
                x = np.random.randint(10**(1 + np.random.randint(8)))
            elif k == 'n':
                x = np.random.randint(10)
            args.append(x)
        return tuple(args)

    def random_vars(spec):
        while True:
            yield random_args(spec)

    # all-real
    args = random_vars(spec)
    for k in range(repeat):
        p = args.next()
        v1, v2 = func(*p)
        v1 = complex(v1)
        v2 = complex(v2)
        if not np.isfinite(v1) or not np.isfinite(v2):
            continue
        sc = 1
        if np.isfinite(np.abs(v1)) and np.abs(v1) > sc:
            sc = abs(v1)
        if np.isfinite(np.abs(v2)) and np.abs(v2) > sc:
            sc = abs(v2)
        a1 = v1 / sc
        a2 = v2 / sc
        assert abs(a1 - a2) < tol, (p, v1, v2)

def test_cos():
    cos = np.cos
    sin = np.sin
    _check_functional(lambda a, b: (cos(a+b), (cos(a)*cos(b)-sin(a)*sin(b))))

def test_struve():
    H = special.struve
    spi = np.sqrt(np.pi)
    gamma = special.gamma

    _check_functional(lambda n, z: (
        H(n,z),
        2*(n+1)/z*H(n+1,z) - H(n+2,z) + 2**(-n-1)*z**(n+1)/spi/gamma(n+5./2)),
                      'iR')

def test_gamma():
    gamma = special.gamma
    _check_functional(lambda z: (gamma(z), gamma(z-1)*(z-1)))
