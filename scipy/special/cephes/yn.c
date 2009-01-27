/*							yn.c
 *
 *	Bessel function of second kind of integer order
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, yn();
 * int n;
 *
 * y = yn( n, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns Bessel function of order n, where n is a
 * (possibly negative) integer.
 *
 * The function is evaluated by forward recurrence on
 * n, starting with values computed by the routines
 * y0() and y1().
 *
 * If n = 0 or 1 the routine for y0 or y1 is called
 * directly.
 *
 *
 *
 * ACCURACY:
 *
 *
 *                      Absolute error, except relative
 *                      when y > 1:
 * arithmetic   domain     # trials      peak         rms
 *    DEC       0, 30        2200       2.9e-16     5.3e-17
 *    IEEE      0, 30       30000       3.4e-15     4.3e-16
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * yn singularity   x = 0              MAXNUM
 * yn overflow                         MAXNUM
 *
 * Spot checked against tables for x, n between 0 and 100.
 *
 */

/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
*/

#include "mconf.h"
#include "bessel_factors.h"
#ifdef ANSIPROT
extern double y0(double);
extern double y1(double);
extern double log(double);
#else
double y0(), y1(), log();
#endif
extern double MAXNUM, MAXLOG, INFINITY, PI, MACHEP, INFINITY;
#ifdef NANS
extern double NAN;
#endif

static double yv_asymptotic_debye(double v, double x);


/* Bessel function of noninteger order */
double yv(double v, double x)
{
     double y, t;
     int n;
     
     if (fabs(v) > 20 && (fabs(x/v) > 2 || fabs(x/v) < 0.5)) {
	  return yv_asymptotic_debye(v, x);
     }
     
     y = floor(v);
     if (y == v) {
	  n = v;
	  y = yn(n, x);
	  return y;
     }

     t = PI * v;
     y = (cos(t) * jv(v, x) - jv(-v, x)) / sin(t);
     return y;
}


/* Bessel function of integer order */
double yn(int n, double x)
{
    double an, anm1, anm2, r;
    int k, sign;

    if (n < 0) {
	n = -n;
	if ((n & 1) == 0)	/* -1**n */
	    sign = 1;
	else
	    sign = -1;
    } else
	sign = 1;


    if (n == 0)
	return (sign * y0(x));
    if (n == 1)
	return (sign * y1(x));

    /* test for domain */
    
    if (x == 0.0) {
	mtherr("yn", SING);
	return -INFINITY;
    } else if (x < 0.0) {
	mtherr("yn", DOMAIN);
	return NAN;
    }

    /* for large `n`, use the debye expansion */
    
    if (n > 20 && (fabs(x/n) > 2 || fabs(x/n) < 0.5)) {
        return yv_asymptotic_debye(n, x);
    }

    /* otherwise, forward recurrence on n */

    anm2 = y0(x);
    anm1 = y1(x);
    k = 1;
    r = 2 * k;
    do {
	an = r * anm1 / x - anm2;
	anm2 = anm1;
	anm1 = an;
	r += 2.0;
	++k;
    }
    while (k < n);


    return (sign * an);
}


/* Compute Yv from (AMS5 9.3.8 + 9.3.7), (AMS5 9.3.15 + 9.3.16),
 * an asymptotic expansion for large or small |x/v| and large |v|
 *
 * XXX: 
 */
static double yv_asymptotic_debye(double v, double x)
{
    double y_prefactor, j_prefactor;
    double t, t2;
    double sum_re, sum_im, z;
    int k, n;
    int sign = 1;
    int aye = 1;
    double tanh_a, a, tan_b, b, psi, tmp, y_val, j_val;
    double divisor, term;

    if (v < 0) {
	/* Negative v; compute J_{-v} and J_{-v} and use (AMS 9.1.6) */
	sign = -1;
	v = -v;
    }

    z = x/v;

    if (fabs(z) <= 1) {
        aye = 1;
        a = acosh(1/z);
        tanh_a = (1 + z)*sqrt((1-z)/(1+z));

	psi = 0;
        j_prefactor = +1/sqrt(2*PI*v*tanh_a) * exp(v*(tanh_a - a));
        y_prefactor = -1/sqrt(.5*PI*v*tanh_a) * exp(-v*(tanh_a - a));
        
        /* Only real numbers in Debye expansion */
        aye = 1;
        t = 1/tanh_a;
        t2 = t*t;
    } else {
        /* The Debye expansion contains complex numbers; we fake them */
        b = acos(1/z);
        tan_b = z * sqrt(1 - 1.0/z/z);

        psi = v*(tan_b - b) - PI/4;
        j_prefactor = 1/sqrt(.5*PI*v*tan_b);
        y_prefactor = 1/sqrt(.5*PI*v*tan_b);

        /* Fake the complex numbers in the debye expansion */
        aye = -1;
        t = 1/tan_b;
        t2 = -t*t;
    }

    if (y_prefactor == INFINITY || y_prefactor == -INFINITY)
	return y_prefactor;

    divisor = v;

    sum_re = 1;
    sum_im = 0;

    for (n = 1; n < N_UFACTORS; ++n) {
	/* Evaluate u_k(t) with Horner's scheme;
	 * (using the knowledge about which coefficients are zero)
	 *
	 * Note how the appropriate definitions of t2 and t count
	 * the real and imaginary parts, as the u polynomials have either
	 * all terms even or all terms odd.
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

        if (n % 2 == 0) {
            sum_re += term;
        } else {
            sum_im += term;
        }

	/* Check convergence */
	if (fabs(term) < MACHEP)
	    break;

	divisor *= v;
    }

    if (fabs(term) > 1e-3*fabs(sum_re)) {
	/* Didn't converge */
	mtherr("yv_asymptotic_uniform", TLOSS);
    }
    if (fabs(term) > MACHEP*fabs(sum_re)) {
	/* Some precision lost */
	mtherr("yv_asymptotic_uniform", PLOSS);
    }

    /* Final result */
    if (aye == 1) {
        y_val = y_prefactor * (sum_re - sum_im);
        j_val = j_prefactor * (sum_re + sum_im);
    } else {
        y_val = y_prefactor * (sum_re*sin(psi) - sum_im*cos(psi));
        j_val = j_prefactor * (sum_re*cos(psi) + sum_im*sin(psi));
    }

    if (sign == -1) {
	/* (AMS 9.1.6) */
	return y_val * cos(PI*v) + j_val * sin(PI*v);
    }
    return y_val;
}

