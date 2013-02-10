/* dump_gsl.c -- Dump special function values to a file.
 *
 * Copyright (C) 2013 Pauli Virtanen
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_sf.h>

static volatile int nan_seen = 0;
static void error_handler(const char *reason, const char *file, int line, int gsl_errno)
{
    nan_seen = 1;
}

#define CALL_SF_FUNC(res, func, args...)         \
    do { nan_seen = 0; func(args, &(res)); if (nan_seen) (res).val = NAN; } while (0)


void dump_mathieu()
{
    gsl_sf_result res_1, res_2, res_3, res_4;
    int jm, jq, jx;
    FILE *f;

    int values_m[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 150, 2000, 20000};
    int nvalues_m = sizeof(values_m) / sizeof(int);

    int values_m2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50};
    int nvalues_m2 = sizeof(values_m2) / sizeof(int);

    double values_q[] = {
        +1000.9, +100.2, +10, +2.5, +1.0, 1e-3+1e-14, +1e-5, +0.5e-7, +1e-9,
        -1000.9, -100.2, -10, -2.5, -1.0, 1e-3+1e-14, -1e-5, -0.5e-7, -1e-9,
        0
    };
    int nvalues_q = sizeof(values_q) / sizeof(double);
    
    double *values_x = values_q;
    int nvalues_x = nvalues_q;

    /* mathieu_a & mathieu_b */
    f = fopen("mathieu_ab.txt", "w");
    fprintf(f, "# m  q  x  a_m(q)  b_m(q)\n");
    for (jm = 0; jm < nvalues_m; ++jm) {
        for (jq = 0; jq < nvalues_q; ++jq) {
            CALL_SF_FUNC(res_1, gsl_sf_mathieu_a, values_m[jm], values_q[jq]);
            CALL_SF_FUNC(res_2, gsl_sf_mathieu_b, values_m[jm], values_q[jq]);
            fprintf(f, "%d %.22g %.22g %.22g\n", values_m[jm], values_q[jq], res_1.val, res_2.val);
            fflush(f);
        }
    }
    fclose(f);

    /* mathieu_ce & mathieu_se */
    f = fopen("mathieu_ce_se.txt", "w");
    fprintf(f, "# m  q  x  ce_m(q,x)  se_m(q,x)\n");
    for (jm = 0; jm < nvalues_m2; ++jm) {
        for (jq = 0; jq < nvalues_q; ++jq) {
            for (jx = 0; jx < nvalues_x; ++jx) {

                if (abs(values_q[jq]) > 50) {
                    /* Bogus results, GSL bug (version 1.15)?
                     * vs. Mathematica
                     */
                    continue;
                }
                if (values_m2[jm] == 4 && fabs(values_q[jq]) < 1e-4) {
                    /* Bogus results, GSL bug (version 1.15)?
                     * vs. Mathematica
                     */
                    continue;
                }

                CALL_SF_FUNC(res_1, gsl_sf_mathieu_ce, values_m2[jm], values_q[jq], values_x[jx]);
                CALL_SF_FUNC(res_2, gsl_sf_mathieu_se, values_m2[jm], values_q[jq], values_x[jx]);
                fprintf(f, "%d %.22g %.22g %.22g %.22g\n", values_m2[jm], values_q[jq],
                        values_x[jx], res_1.val, res_2.val);
                fflush(f);
            }
        }
    }
    fclose(f);

    /* mathieu_mc & mathieu_ms */
    f = fopen("mathieu_mc_ms.txt", "w");
    fprintf(f, "# m  q  x  Mc_m(1,q,x)  Ms_m(1,q,x)  Mc_m(2,q,x)  Ms_m(2,q,x)\n");
    for (jm = 0; jm < nvalues_m2; ++jm) {
        for (jq = 0; jq < nvalues_q; ++jq) {
            for (jx = 0; jx < nvalues_x; ++jx) {

                if (fabs(values_q[jq]) < 0.1) {
                    /* Bogus results, GSL bug (version 1.15)?
                     */
                    continue;
                }

                if (fabs(values_x[jx]) > 20) {
                    /* Bogus results, GSL bug (version 1.15)?
                     */
                    continue;
                }

                if (values_m2[jm] >= 10) {
                    /* Bogus results, GSL bug (version 1.15)?
                     */
                    continue;
                }

                if (values_x[jx] < 0) {
                    /* Bogus results, GSL bug (version 1.15)?
                     */
                    continue;
                }

                CALL_SF_FUNC(res_1, gsl_sf_mathieu_Mc, 1, values_m2[jm], values_q[jq], values_x[jx]);
                CALL_SF_FUNC(res_2, gsl_sf_mathieu_Ms, 1, values_m2[jm], values_q[jq], values_x[jx]);
                CALL_SF_FUNC(res_3, gsl_sf_mathieu_Mc, 2, values_m2[jm], values_q[jq], values_x[jx]);
                CALL_SF_FUNC(res_4, gsl_sf_mathieu_Ms, 2, values_m2[jm], values_q[jq], values_x[jx]);
                fprintf(f, "%d %.22g %.22g %.22g %.22g %.22g %.22g\n", values_m2[jm], values_q[jq],
                        values_x[jx], res_1.val, res_2.val, res_3.val, res_4.val);
                fflush(f);
            }
        }
    }
    fclose(f);
}

int main(int argc, char *argv[])
{
    gsl_set_error_handler(error_handler);

    dump_mathieu();
    
    exit(0);
}
