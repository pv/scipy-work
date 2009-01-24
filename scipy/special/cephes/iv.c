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
    double mu, mup;
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


/* Uniform asymptotic expansion factors, (AMS5 9.3.9; AMS5 9.3.10)
 *
 * Computed with:
 * --------------------
   import numpy as np
   t = np.poly1d([1,0])
   def up1(p):
       return .5*t*t*(1-t*t)*p.deriv() + 1/8. * ((1-5*t*t)*p).integ()
   us = [np.poly1d([1])]
   for k in range(10):
       us.append(up1(us[-1]))
   n = us[-1].order
   for p in us:
       print "{" + ", ".join(["0"]*(n-p.order) + map(repr, p)) + "},"
   print "N_UFACTORS", len(us)
   print "N_UFACTOR_TERMS", us[-1].order + 1
 * --------------------
 */
#define N_UFACTORS 11
#define N_UFACTOR_TERMS 31
static const double asymptotic_ufactors[N_UFACTORS][N_UFACTOR_TERMS] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, -0.20833333333333334, 0.0, 0.125, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0.3342013888888889, 0.0, -0.40104166666666669, 0.0, 0.0703125, 0.0,
     0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -1.0258125964506173, 0.0, 1.8464626736111112, 0.0,
     -0.89121093750000002, 0.0, 0.0732421875, 0.0, 0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     4.6695844234262474, 0.0, -11.207002616222995, 0.0, 8.78912353515625,
     0.0, -2.3640869140624998, 0.0, 0.112152099609375, 0.0, 0.0, 0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28.212072558200244, 0.0,
     84.636217674600744, 0.0, -91.818241543240035, 0.0, 42.534998745388457,
     0.0, -7.3687943594796312, 0.0, 0.22710800170898438, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 212.5701300392171, 0.0,
     -765.25246814118157, 0.0, 1059.9904525279999, 0.0,
     -699.57962737613275, 0.0, 218.19051174421159, 0.0,
     -26.491430486951554, 0.0, 0.57250142097473145, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, -1919.4576623184068, 0.0,
     8061.7221817373083, 0.0, -13586.550006434136, 0.0, 11655.393336864536,
     0.0, -5305.6469786134048, 0.0, 1200.9029132163525, 0.0,
     -108.09091978839464, 0.0, 1.7277275025844574, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 20204.291330966149, 0.0, -96980.598388637503, 0.0,
     192547.0012325315, 0.0, -203400.17728041555, 0.0, 122200.46498301747,
     0.0, -41192.654968897557, 0.0, 7109.5143024893641, 0.0,
     -493.915304773088, 0.0, 6.074042001273483, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0},
    {0, 0, 0, -242919.18790055133, 0.0, 1311763.6146629769, 0.0,
     -2998015.9185381061, 0.0, 3763271.2976564039, 0.0,
     -2813563.2265865342, 0.0, 1268365.2733216248, 0.0,
     -331645.17248456361, 0.0, 45218.768981362737, 0.0,
     -2499.8304818112092, 0.0, 24.380529699556064, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0},
    {3284469.8530720375, 0.0, -19706819.11843222, 0.0, 50952602.492664628,
     0.0, -74105148.211532637, 0.0, 66344512.274729028, 0.0,
     -37567176.660763353, 0.0, 13288767.166421819, 0.0,
     -2785618.1280864552, 0.0, 308186.40461266245, 0.0,
     -13886.089753717039, 0.0, 110.01714026924674, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0}
};


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
	    term += asymptotic_ufactors[n][k];
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
