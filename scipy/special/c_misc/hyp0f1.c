/*
 * Compute the hypergeometric 0F1 function.
 *
 * References
 * ----------
 * [1] NIST Digital Library of Mathematical Functions
 *     http://dlmf.nist.gov/16
 */

/*
 * Copyright (C) 2013  Pauli Virtanen
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * b. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * c. Neither the name of Enthought nor the names of the SciPy Developers
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <math.h>

#include "cephes.h"
#include "amos_wrappers.h"
#include "misc.h"

#include "double2.h"

#define MAXITER 10000
#define SUM_EPS 1e-100   /* be sure we are in the tail of the sum */
#define GOOD_EPS 1e-12
#define ACCEPTABLE_EPS 1e-7
#define ACCEPTABLE_ATOL 1e-300

#define MIN(a, b) ((a) < (b) ? (a) : (b))

double hyp_0f1(double b, double z)
{
    int n;
    double value[1], err[1];

    /* Try power series */
    value[0] = hyp_0f1_power_series(b, z, &err[0]);
    if (err[0] < GOOD_EPS * fabs(value[0])) {
        return value[0];
    }

    /* Return the best of the three, if it is acceptable */
    n = 0;
    if (err[n] < ACCEPTABLE_EPS * fabs(value[n]) || err[n] < ACCEPTABLE_ATOL) {
        return value[n];
    }

    /* Failure */
    sf_error("hyp0f1", SF_ERROR_NO_RESULT, "total loss of precision");
    return NPY_NAN;
}


/*
 * Power series for hyp0f1
 * http://dlmf.nist.gov/16
 */
double hyp_0f1_power_series(double b, double z, double *err)
{
    int n, sgn;
    double term, sum, maxterm;
    double xterm, xsum;
    double2_t cterm, csum, cdiv, cz, cb, cn, ctmp, ctmp2;

    term = 1.0;
    sum = term;
    maxterm = 0;

    xterm = term;
    xsum = xterm;

    double2_init(&cterm, term);
    double2_init(&csum, sum);

    double2_init(&cz, z);
    double2_init(&cb, b);

    for (n = 0; n < MAXITER; ++n) {
        /* cterm *= z / (b + n) / (n + 1)  */
        double2_mul(&cterm, &cz, &cterm);

        double2_init(&cn, n);

        double2_add(&cb, &cn, &ctmp);
        double2_div(&cterm, &ctmp, &cterm);

        double2_init(&ctmp, n + 1);
        double2_div(&cterm, &ctmp, &cterm);

        /* csum += cterm */
        double2_add(&csum, &cterm, &csum);

        term = double2_double(&cterm);
        sum = double2_double(&csum);

        if (fabs(term) > maxterm) {
            maxterm = fabs(term);
        }
        if (fabs(term) < SUM_EPS * fabs(sum) || term == 0 || !npy_isfinite(sum)) {
            break;
        }
    }

    *err = fabs(term) + fabs(maxterm) * 1e-24;
    return sum;
}
