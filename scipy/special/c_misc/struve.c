/*
 * Compute the Struve function.
 *
 * Notes
 * -----
 *
 * We use three expansions for the Struve function discussed in [1]:
 *
 * - power series
 * - expansion in Bessel functions
 * - asymptotic large-z expansion
 *
 * Rounding errors are estimated based on the largest terms in the sums.
 *
 * (i)
 *
 * Looking at the error in the asymptotic expansion, one finds that
 * it's not worth trying it out unless |z| ~> 0.7 * |v| + 12.
 *
 * (ii)
 *
 * The Bessel function expansion tends to fail for |z| >~ |v| and is not tried
 * there.
 *
 * For Struve H it covers the quadrant v > z where the power series tends to
 * fail to produce reasonable results due to loss of precision.
 *
 * (iii)
 *
 * The three expansions together cover for Struve H the region z > 0, v real.
 *
 * (iv)
 *
 * For Struve L there remains a difficult region around v < 0, z ~ -0.7 v >> 1,
 * where none of the three expansions converges.
 *
 * This implementation returns NAN in this region.
 *
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
#define SUM_EPS 1e-16
#define GOOD_EPS 1e-12
#define ACCEPTABLE_EPS 1e-7
#define ACCEPTABLE_ATOL 1e-300

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static double struve_power_series(double v, double x, int is_h, double *err);
static double struve_asymp_large_z(double v, double z, int is_h, double *err);
static double struve_bessel_series(double v, double z, int is_h, double *err);
static double bessel_y(double v, double x);
static double bessel_i(double v, double x);
static double bessel_j(double v, double x);
static double struve_hl(double v, double x, int is_h);
extern double polevl ( double x, void *P, int N );

double struve_h(double v, double z)
{
    return struve_hl(v, z, 1);
}

double struve_l(double v, double z)
{
    return struve_hl(v, z, 0);
}

static double struve_hl(double v, double z, int is_h)
{
    double value[3], err[3];
    int n;

    if (z < 0) {
        return NPY_NAN;
    }
    else if (z == 0) {
        if (v < -1) {
            return gammasgn(v + 1.5) * NPY_INFINITY;
        }
        else if (v == -1) {
            return 2 / sqrt(M_PI) / Gamma(0.5);
        }
        else {
            return 0;
        }
    }

    n = -v - 0.5;
    if (n == -v - 0.5 && n > 0) {
        if (is_h) {
            return (n % 2 == 0 ? 1 : -1) * bessel_j(n + 0.5, z);
        }
        else {
            return bessel_i(n + 0.5, z);
        }
    }

    if (z >= 0.7*fabs(v) + 12) {
        /* Worth trying the asymptotic expansion */
        value[0] = struve_asymp_large_z(v, z, is_h, &err[0]);
        if (err[0] < GOOD_EPS * fabs(value[0])) {
            return value[0];
        }
    }
    else {
        err[0] = NPY_INFINITY;
    }

    value[1] = struve_power_series(v, z, is_h, &err[1]);
    if (err[1] < GOOD_EPS * fabs(value[1])) {
        return value[1];
    }

    if (fabs(z) < fabs(v) + 20) {
        value[2] = struve_bessel_series(v, z, is_h, &err[2]);
        if (err[2] < GOOD_EPS * fabs(value[2])) {
            return value[2];
        }
    }
    else {
        err[2] = NPY_INFINITY;
    }

    /* Return the best of the three, if it is acceptable */
    n = 0;
    if (err[1] < err[n]) n = 1;
    if (err[2] < err[n]) n = 2;
    if (err[n] < ACCEPTABLE_EPS * fabs(value[n]) || err[n] < ACCEPTABLE_ATOL) {
        return value[n];
    }

    /* Maybe it really is an overflow? */
    if (!npy_isfinite(value[1])) {
        sf_error("struve", SF_ERROR_OVERFLOW, "overflow in series");
        return value[1];
    }

    sf_error("struve", SF_ERROR_NO_RESULT, "total loss of precision");
    return NPY_NAN;
}


/*
 * Power series for Struve H and L
 * http://dlmf.nist.gov/11.2.1
 *
 * Starts to converge roughly at |n| > |z|
 */
static double struve_power_series(double v, double z, int is_h, double *err)
{
    int n, sgn;
    double term, sum, maxterm;

    if (is_h) {
        sgn = -1;
    }
    else {
        sgn = 1;
    }
    
    term = 2 / sqrt(M_PI) * exp(-lgam(v + 1.5) + (v + 1)*log(z/2)) * gammasgn(v + 1.5);
    sum = term;
    maxterm = 0;

    for (n = 0; n < MAXITER; ++n) {
        term *= sgn * z*z / (3 + 2*n) / (3 + 2*n + 2*v);
        sum += term;
        if (fabs(term) > maxterm) {
            maxterm = fabs(term);
        }
        if (fabs(term) < SUM_EPS * fabs(sum) || term == 0 || !npy_isfinite(sum)) {
            break;
        }
    }
    *err = fabs(term) + fabs(maxterm) * 1e-16;
    return sum;
}


/*
 * Bessel series
 * http://dlmf.nist.gov/11.4.19
 */
static double struve_bessel_series(double v, double z, int is_h, double *err)
{
    int n, sgn;
    double term, cterm, sum, maxterm;

    if (is_h && v < 0) {
        /* Works less reliably in this region */
        *err = NPY_INFINITY;
        return NPY_NAN;
    }
    
    sum = 0;
    maxterm = 0;

    cterm = sqrt(z / (2*M_PI));

    for (n = 0; n < MAXITER; ++n) {
        if (is_h) {
            term = cterm * bessel_j(n + v + 0.5, z) / (n + 0.5);
            cterm *= z/2 / (n + 1);
        }
        else {
            term = cterm * bessel_i(n + v + 0.5, z) / (n + 0.5);
            cterm *= -z/2 / (n + 1);
        }
        sum += term;
        if (fabs(term) > maxterm) {
            maxterm = fabs(term);
        }
        if (fabs(term) < SUM_EPS * fabs(sum) || term == 0 || !npy_isfinite(sum)) {
            break;
        }
    }

    *err = fabs(term) + fabs(maxterm) * 1e-16;
    return sum;
}


/*
 * Large-z expansion for Struve H and L
 * http://dlmf.nist.gov/11.6.1
 */
static double struve_asymp_large_z(double v, double z, int is_h, double *err)
{
    int n, sgn, maxiter;
    double term, sum, maxterm;
    double m;

    if (is_h) {
        sgn = -1;
    }
    else {
        sgn = 1;
    }

    /* Asymptotic expansion divergenge point */
    m = z/2;
    if (m <= 0) {
        maxiter = 0;
    }
    else if (m > MAXITER) {
        maxiter = MAXITER;
    }
    else {
        maxiter = (int)m;
    }
    if (maxiter == 0) {
        *err = NPY_INFINITY;
        return NPY_NAN;
    }

    /* Evaluate sum */
    term = -sgn / sqrt(M_PI) * exp(-lgam(v + 0.5) + (v - 1) * log(z/2)) * gammasgn(v + 0.5);
    sum = term;
    maxterm = 0;

    for (n = 0; n < maxiter; ++n) {
        term *= sgn * (1 + 2*n) * (1 + 2*n - 2*v) / (z*z);
        sum += term;
        if (fabs(term) > maxterm) {
            maxterm = fabs(term);
        }
        if (fabs(term) < SUM_EPS * fabs(sum) || term == 0 || !npy_isfinite(sum)) {
            break;
        }
    }

    if (is_h) {
        sum += bessel_y(v, z);
    }
    else {
        sum += bessel_i(v, z);
    }

    *err = fabs(term) + fabs(maxterm) * 1e-16;

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

static double bessel_j(double v, double x)
{
    return cbesj_wrap_real(v, x);
}
