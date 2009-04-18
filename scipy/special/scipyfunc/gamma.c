/*
 * XXX: placeholder; borrow implementation from Cephes
 */

#include "scipyfunc.h"
#include "cephes_protos.h"

double scf_gamma(double x)
{
    return Gamma(x);
}

float scf_gammaf(float x)
{
    return (float)Gamma((double)x);
}

npy_longdouble scf_gammal(npy_longdouble x)
{
    return (npy_longdouble)Gamma((double)x);
}
