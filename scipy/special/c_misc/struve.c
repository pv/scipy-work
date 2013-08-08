/*
 * Compute the Struve function.
 *
 * Notes
 * -----
 *
 * We use three expansions for the Struve function discussed in [1]:
 *
 * - power series
 * - asymptotic large-z expansion
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
#define ACCEPTABLE_EPS 1e-9

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static double struve_power_series(double v, double x, int is_h, double *err);
static double struve_asymp_large_z(double v, double z, int is_h, double *err);
static double bessel_y(double v, double x);
static double bessel_i(double v, double x);
static double bessel_j(double v, double x);
static double struve_asymp_large_z_log_error_est(double v, double z, int is_h);
static double struve_hl(double v, double x, int is_h);

double struve_h(double v, double x)
{
    return struve_hl(v, x, 1);
}

double struve_l(double v, double x)
{
    return struve_hl(v, x, 0);
}

static double struve_hl(double v, double x, int is_h)
{
    double value, err;
    int n;

    if (x < 0) {
        return NPY_NAN;
    }
    else if (x == 0) {
        if (v <= -1) {
            return NPY_INFINITY;
        }
        else {
            return 0;
        }
    }

    n = -v - 0.5;
    if (n == -v - 0.5 && n > 0) {
        if (is_h) {
            return (n % 2 == 0 ? 1 : -1) * bessel_j(n + 0.5, x);
        }
        else {
            return bessel_i(n + 0.5, x);
        }
    }

    //if (struve_asymp_large_z_log_error_est(v, x, is_h) < log(ACCEPTABLE_EPS) +
    //5) {
    value = struve_asymp_large_z(v, x, is_h, &err);
    if (err < ACCEPTABLE_EPS * fabs(value)) {
        return value;
    }

    value = struve_power_series(v, x, is_h, &err);
    if (err < ACCEPTABLE_EPS * fabs(value)) {
        return value;
    }

    sf_error("struve", SF_ERROR_NO_RESULT, "total loss of precision");
    return 1234;
}


/*
 * The large-z asymptotic series [1] converges for |z| >> 1 provided
 *
 *     |z| > f(v)
 *
 * An approximat condition follows from the asymptotic series error
 * estimate. The terms in the asymptotic series stop decreasing roughly at
 *
 *     k ~ m = z/2
 *
 * Using the term T_m as the error estimate and requiring |T_m| < 1e-9 |T_0|
 * gives the approximate threshold expression used below.
 *
 */
static double struve_asymp_large_z_log_error_est(double v, double z, int is_h)
{
    double log_error, log_term0, log_bessel;

    if (z < 1) {
        /* The error estimate is not valid, see [1]:(11.6.1) */
        return NPY_INFINITY;
    }

    if (v == 0) {
        log_error = - z*log(fabs(z))/2 + z*log(z)/2 - z;
    }
    else {
        log_error = v*log(fabs(v)) - v*log(fabs(2*v - z)) + v*log(2) - z*log(fabs(z))/2 + z*log(fabs(2*v - z))/2 - z;
    }
    log_term0 = -log(sqrt(M_PI)) - lgam(v + 0.5) + (v - 1) * log(z/2);

    if (is_h) {
        log_bessel = -.5*log(z);
    }
    else {
        log_bessel = log(fabs(bessel_i(v, z)));
    }
    if (log_bessel > log_term0) {
        log_error = MIN(log_error, log_term0 - log_bessel);
    }

    return log_error;
}

/*
 * Large-z expansion for Struve H and L
 * http://dlmf.nist.gov/11.6.1
 */
static double struve_asymp_large_z(double v, double z, int is_h, double *err)
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
    m = v + z/2;
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
    for (n = 0; n < maxiter; ++n) {
        term *= sgn * (1 + 2*n) * (1 + 2*n - 2*v) / (z*z);
        sum += term;
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

    *err = fabs(term);

    return sum;
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
    double term, sum;

    if (is_h) {
        sgn = -1;
    }
    else {
        sgn = 1;
    }
    
    sum = 0;
    term = 2 / sqrt(M_PI) * exp(-lgam(v + 1.5) + (v + 1)*log(z/2)) * gammasgn(v + 1.5);

    for (n = 0; n < MAXITER; ++n) {
        sum += term;
        term *= sgn * z*z / (3 + 2*n) / (3 + 2*n + 2*v);
        if (fabs(term) < SUM_EPS * fabs(sum) || term == 0 || !npy_isfinite(sum)) {
            break;
        }
    }
    *err = fabs(term);
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
