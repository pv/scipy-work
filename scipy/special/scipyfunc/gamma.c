/*
 * XXX: placeholder; borrow implementation from Cephes
 */

#include "scipyfunc.h"

extern double gamma ( double );

double scf_gamma(double x)
{
    return gamma(x);
}

float scf_gammaf(float x)
{
    return (float)gamma((double)x);
}

npy_longdouble scf_gammal(npy_longdouble x)
{
    return (npy_longdouble)gamma((double)x);
}
