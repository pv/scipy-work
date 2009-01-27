/*							iv.c
 *
 *	Modified Bessel function of noninteger order
 *
 *
 *
 * SYNOPSIS:
 *
 * double v, x, y, iv();
 *
 * y = iv( v, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns modified Bessel function of order v of the
 * argument.  If x is negative, v must be integer valued.
 *
 * The function is defined as Iv(x) = Jv( ix ).  It is
 * here computed in terms of the confluent hypergeometric
 * function, according to the formula
 *
 *              v  -x
 * Iv(x) = (x/2)  e   hyperg( v+0.5, 2v+1, 2x ) / gamma(v+1)
 *
 * If v is a negative integer, then v is replaced by -v.
 *
 *
 * ACCURACY:
 *
 * Tested at random points (v, x), with v between 0 and
 * 30, x between 0 and 28.
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    DEC       0,30          2000      3.1e-15     5.4e-16
 *    IEEE      0,30         10000      1.7e-14     2.7e-15
 *
 * Accuracy is diminished if v is near a negative integer.
 *
 * See also hyperg.c.
 *
 */
/*							iv.c	*/
/*	Modified Bessel function of noninteger order		*/
/* If x < 0, then v must be an integer. */


/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1988, 2000 by Stephen L. Moshier
*/

#include "mconf.h"
#include "bessel_factors.h"
#ifdef ANSIPROT
extern double hyperg(double, double, double);
extern double exp(double);
extern double gamma(double);
extern double log(double);
extern double fabs(double);
extern double floor(double);
#else
double hyperg(), exp(), gamma(), log(), fabs(), floor();
#endif
extern double MACHEP, MAXNUM, NAN, PI, INFINITY;

static double iv_asymptotic(double v, double x);
static double iv_asymptotic_uniform(double v, double x);

double iv(double v, double x)
{
    int sign;
    double t, vp, ax, res;

    /* If v is a negative integer, invoke symmetry */
    t = floor(v);
    if (v < 0.0) {
	if (t == v) {
	    v = -v;		/* symmetry */
	    t = -t;
	}
    }
    /* If x is negative, require v to be an integer */
    sign = 1;
    if (x < 0.0) {
	if (t != v) {
	    mtherr("iv", DOMAIN);
	    return (NAN);
	}
	if (v != 2.0 * floor(v / 2.0))
	    sign = -1;
    }

    /* Avoid logarithm singularity */
    if (x == 0.0) {
	if (v == 0.0)
	    return (1.0);
	if (v < 0.0) {
	    mtherr("iv", OVERFLOW);
	    return (MAXNUM);
	} else
	    return (0.0);
    }

    ax = fabs(x);

    /* Uniform asymptotic expansion for large orders */
    if (fabs(v) > 15) {
	/* Note: the treshold here is chosen so that
	 *       the hyperg method and the asymptotic series below work for all
	 *       remaining x, v.
	 */
	return sign * iv_asymptotic_uniform(v, ax);
    }

    /* Asymptotic expansion */
    if (ax > 30) {
	return sign * iv_asymptotic(v, ax);
    }

    /* Elsewhere: use the relation to hypergeometic function */

    if (v < 0 && floor(v + 0.5) == v + 0.5) {
        /* Avoid hyperg singularity */
	v -= 2 * MACHEP * v;
    }

    t = v * log(0.5 * ax) - ax;
    t = sign * exp(t) / gamma(v + 1.0);
    vp = v + 0.5;
    res = hyperg(vp, 2.0 * vp, 2.0 * ax);
    return (t * res);
}


/* Compute Iv from (AMS5 9.7.1), asymptotic expansion for large |z|
 * Iv ~ exp(x)/sqrt(2 pi x) ( 1 + (4*v*v-1)/8x + (4*v*v-1)(4*v*v-9)/8x/2! + ...)
 */
static double iv_asymptotic(double v, double x)
{
    double mu;
    double sum, term, prefactor, factor;
    int k;

    prefactor = exp(x) / sqrt(2 * PI * x);

    if (prefactor == INFINITY)
	return prefactor;

    mu = 4 * v * v;
    sum = 1.0;
    term = 1.0;
    k = 1;

    do {
	factor = (mu - (2 * k - 1) * (2 * k - 1)) / (8 * x) / k;
	if (k > 100) {
	    /* didn't converge */
	    mtherr("iv(iv_asymptotic)", TLOSS);
	    break;
	}
	term *= -factor;
	sum += term;
	++k;
    } while (fabs(term) > MACHEP * fabs(sum));
    return sum * prefactor;
}


/* Compute Iv from (AMS5 9.7.7 + 9.7.8), asymptotic expansion for large v
 */
static double iv_asymptotic_uniform(double v, double x)
{
    double i_prefactor, k_prefactor;
    double t, t2, eta, z;
    double i_sum, k_sum, term, divisor;
    int k, n;
    int sign = 1;

    if (v < 0) {
	/* Negative v; compute I_{-v} and K_{-v} and use (AMS 9.6.2) */
	sign = -1;
	v = -v;
    }

    z = x / v;
    t = 1 / sqrt(1 + z * z);
    t2 = t * t;
    eta = sqrt(1 + z * z) + log(z / (1 + 1 / t));

    i_prefactor = sqrt(t / (2 * PI * v)) * exp(v * eta);
    i_sum = 1.0;

    if (sign == -1) {
	k_prefactor = sqrt(PI * t / (2 * v)) * exp(-v * eta);
	k_sum = 1.0;
    }

    divisor = v;
    for (n = 1; n < N_UFACTORS; ++n) {
	/* Evaluate u_k(t) with Horner's scheme;
	 * (using the knowledge about which coefficients are zero)
	 */
	term = 0;
	for (k = N_UFACTOR_TERMS - 1 - 3 * n;
	     k < N_UFACTOR_TERMS - n; k += 2) {
	    term *= t2;
	    term += bessel_ufactors[n][k];
	}
	for (k = 1; k < n; k += 2)
	    term *= t2;
	if (n % 2 == 1)
	    term *= t;

	/* Sum terms */
	term /= divisor;
	i_sum += term;
	if (sign == -1)
	    k_sum += (n % 2 == 0) ? term : -term;

	/* Check convergence */
	if (fabs(term) < MACHEP)
	    break;

	divisor *= v;
    }

    if (fabs(term) > 1e-3*fabs(i_sum)) {
	/* Didn't converge */
	mtherr("iv(iv_asymptotic_uniform)", TLOSS);
    }
    if (fabs(term) > MACHEP*fabs(i_sum)) {
	/* Some precision lost */
	mtherr("iv(iv_asymptotic_uniform)", PLOSS);
    }

    if (sign == -1) {
	/* (AMS 9.6.2) */
	return i_prefactor * i_sum +
	    (2 / PI) * sin(PI * v) * k_prefactor * k_sum;
    }
    return i_prefactor * i_sum;
}
