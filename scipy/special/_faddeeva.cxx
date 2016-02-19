#include <complex>
#include <limits>

#include "_faddeeva.h"

using namespace std;

EXTERN_C_START

npy_cdouble faddeeva_w(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    std::complex<double> w = Faddeeva::w(z);
    return npy_cpack(real(w), imag(w));
}

npy_cdouble faddeeva_erf(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erf(z);
    return npy_cpack(real(w), imag(w));
}

npy_cdouble faddeeva_erfc(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfc(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfcx(double x)
{
    return Faddeeva::erfcx(x);
}

npy_cdouble faddeeva_erfcx_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfcx(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_erfi(double x)
{
    return Faddeeva::erfi(x);
}

npy_cdouble faddeeva_erfi_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::erfi(z);
    return npy_cpack(real(w), imag(w));
}

double faddeeva_dawsn(double x)
{
    return Faddeeva::Dawson(x);
}

npy_cdouble faddeeva_dawsn_complex(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    complex<double> w = Faddeeva::Dawson(z);
    return npy_cpack(real(w), imag(w));
}

/*
 * A wrapper for a normal CDF for complex argument
 */

npy_cdouble faddeeva_ndtr(npy_cdouble zp)
{
    complex<double> z(zp.real, zp.imag);
    z *= NPY_SQRT1_2;
    complex<double> w = 0.5 * Faddeeva::erfc(-z);
    return npy_cpack(real(w), imag(w));
}

/*
 * Logarithm of the normal CDF, with improved accuracy for z->-inf.
 * This is essentially a copy of log_ndtr routine from cephes/ndtr.c
 * 
 * For Re(a) > -20, use the existing ndtr technique and take a log.
 * For Re(a) <= -20, use the asymptotic series of A&S 26.2.12
 * (equivalently, A&S 7.1.23)
 */

npy_cdouble faddeeva_log_ndtr(npy_cdouble zp)
{

    complex<double> a(zp.real, zp.imag), result;
    complex<double> log_LHS,        /* we compute the left hand side of the approx (LHS) in one shot */
        last_total = 0,        /* variable used to check for convergence */
        right_hand_side = 1,    /* includes first term from the RHS summation */
        numerator = 1,        /* numerator for RHS summand */
        denom_factor = 1,    /* use reciprocal for denominator to avoid division */
        denom_cons = 1.0 / (a * a);    /* the precomputed division we use to adjust the denominator */
        double sign = 1;
        long i = 0;

    if (a.real() > -20) {
        return npy_clog(faddeeva_ndtr(zp));
    }
    log_LHS = -0.5 * a * a - log(-a) - 0.5 * log(2 * M_PI);

    while (std::abs(last_total - right_hand_side) > std::numeric_limits<double>::epsilon()) {
        i += 1;
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2.0 * i - 1.0;
        right_hand_side += sign * numerator * denom_factor;
    }
    result = log_LHS + log(right_hand_side);
    return npy_cpack(result.real(), result.imag());
}

EXTERN_C_END
