/*
 * Compute the Struve function.
 *
 * References
 * ----------
 * [1] NIST Digital Library of Mathematical Functions
 *     http://dlmf.nist.gov/11
 */
#include <stdio.h>
#include <math.h>

#include "cephes.h"
#include "amos_wrappers.h"
#include "misc.h"

#define MAXITER 10000
#define SUMEPS 1e-16

static double struve_power_series(double v, double x, int is_h);
static double struve_asymp_large_z(double v, double z, int is_h);
static double bessel_y(double v, double x);
static double bessel_i(double v, double x);

double struve_h(double v, double x)
{
    if (x < 0) {
        return NPY_NAN;
    }
    else if (x == 0) {
        if (v < -1) {
            return NPY_INFINITY;
        }
        else if (v == 1) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else if (fabs(x) > 10.0) {
        return struve_asymp_large_z(v, x, 1);
    }
    else {
        return struve_power_series(v, x, 1);
    }
}

double struve_l(double v, double x)
{
    if (x < 0) {
        return NPY_NAN;
    }
    else if (x == 0) {
        if (v < -1) {
            return NPY_INFINITY;
        }
        else if (v == 1) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else if (fabs(x) > 25.0) {
        return struve_asymp_large_z(v, x, 0);
    }
    else {
        return struve_power_series(v, x, 0);
    }
}

/*
 * Large-z expansion for Struve H and L
 * http://dlmf.nist.gov/11.6.1
 */
static double struve_asymp_large_z(double v, double z, int is_h)
{
    int n, sgn, maxiter;
    double term, sum;
    double m;

    if (is_h) {
        sgn = -1;
    }
    else {
        sgn = 1;
    }

    /* Asymptotic expansion divergenge point */
    m = (sqrt(v*v + z*z) + v - 1)/2;
    if (m < 0) {
        maxiter = 0;
    }
    else if (m > MAXITER) {
        maxiter = MAXITER;
    }
    else {
        maxiter = (int)m;
    }

    /* Evaluate sum */
    sum = 0;
    term = -sgn / sqrt(M_PI) * exp(-lgam(v + 0.5) + (v - 1) * log(z/2)) * gammasgn(v + 0.5);
    for (n = 0; n < maxiter; ++n) {
        sum += term;
        term *= sgn * (1 + 2*n) * (1 + 2*n - 2*v) / (z*z);
        if (fabs(term) < SUMEPS * fabs(sum)) {
            break;
        }
    }

    if (n == maxiter) {
        /* Didn't converge in time */
        return 1234;
    }

    if (is_h) {
        sum += bessel_y(v, z);
    }
    else {
        sum += bessel_i(v, z);
    }

    return sum;
}

/*
 * Power series for Struve H and L
 * http://dlmf.nist.gov/11.2.1
 *
 * Starts to converge roughly at |n| > |z|
 */
static double struve_power_series(double v, double z, int is_h)
{
    int n, sgn;
    double term, sum;

    if (is_h) {
        sgn = -1;
    }
    else {
        sgn = 1;
    }
    
    sum = 0;
    term = 2 / sqrt(M_PI) / Gamma(v + 1.5) * pow(z/2, v + 1);

    for (n = 0; n < MAXITER; ++n) {
        sum += term;
        term *= sgn * z*z / (3 + 2*n) / (3 + 2*n + 2*v);
        if (fabs(term) < SUMEPS * fabs(sum)) {
            break;
        }
    }
    if (n == MAXITER) {
        /* Didn't converge */
        return 1234;
    }
    return sum;
}

static double bessel_y(double v, double x)
{
    return cbesy_wrap_real(v, x);
}

static double bessel_i(double v, double x)
{
    return cephes_iv(v, x);
}
