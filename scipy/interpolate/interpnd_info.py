"""
Here we perform some symbolic computations required for the N-D
interpolation routines in `interpnd.pyx`.

"""
from sympy import *

def _estimate_gradients_2d_global():

    #
    # Compute
    #
    #

    f1, f2, df1, df2, x = symbols(['f1', 'f2', 'df1', 'df2', 'x'])
    c = [f1, (df1 + 3*f1)/3, (df2 + 3*f2)/3, f2]

    w = 0
    for k in range(4):
        w += binomial(3, k) * c[k] * x**k*(1-x)**(3-k)

    wpp = w.diff(x, 2).expand()
    intwpp2 = (wpp**2).integrate((x, 0, 1)).expand()

    A = Matrix([[intwpp2.coeff(df1**2), intwpp2.coeff(df1*df2)/2],
                [intwpp2.coeff(df1*df2)/2, intwpp2.coeff(df2**2)]])

    B = Matrix([[intwpp2.coeff(df1).subs(df2, 0)],
                [intwpp2.coeff(df2).subs(df1, 0)]]) / 2

    print "A"
    print A
    print "B"
    print B
    print "solution"
    print A.inv() * B

    syms = [f1, f2, df1, df2]

    print intwpp2.subs(df1, f2 - f1).subs(df2, f1 - f2).expand()

    AX = Matrix([[intwpp2.coeff(a*b) / (1 if a is b else 2)
                  for b in syms]
                 for a in syms])

    BX = Matrix([[intwpp2.coeff(a).subs(f1, 0).subs(f2, 0).subs(df1, 0).subs(df2, 0)]
                 for a in syms]) / 2

    print "AX"
    print AX
    print "BX"
    print BX

if __name__ == "__main__":
    _estimate_gradients_2d_global()
