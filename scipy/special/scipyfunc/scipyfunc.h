#ifndef __SCIPYFUNC_H_
#define __SCIPYFUNC_H_

#include <Python.h>
#include <numpy/npy_math.h>
#include <math.h>
#include <stdlib.h>

/*
 * Floating-point precision constants
 */

/*
 * XXX: fix these!
 */

#ifndef NAN
#define NAN  NPY_NAN
#endif
#define NANF NPY_NANF
#define NANL NPY_NANL

#ifndef INFINITY
#define INFINITY NPY_INFINITY
#endif
#define INFINITYF NPY_INFINITYF
#define INFINITYL NPY_INFINITYL

#ifndef PZERO
#define PZERO NPY_PZERO
#endif
#define PZEROF NPY_PZEROF
#define PZEROL NPY_PZEROL

#define EPSILON  2.2204460492503131e-16
#define EPSILONF 1.1920929e-07F
#define EPSILONL 1e-19L

#define MAXNUM  1.7976931348623157e+308
#define MAXNUMF 3.4028235e+38F
#define MAXNUML 1.189731495357231765e+4932L

#define MAXLOG  7.08396418532264106224E2
#define MAXLOGF 1e1F
#define MAXLOGL 7.08396418532264106224E2L

/*
 * Mathematical constants
 */

#define EULER  0.577215664901532860606512090082402
#define EULERF 0.577215664901532860606512090082402F
#define EULERL 0.577215664901532860606512090082402L

#define PI NPY_PI
#define PIF NPY_PIf
#define PIL NPY_PIl

/*
 * Error handling
 */
#define DOMAIN          1       /* argument domain error */
#define SING            2       /* argument singularity */
#define OVERFLOW        3       /* overflow range error */
#define UNDERFLOW       4       /* underflow range error */
#define TLOSS           5       /* total loss of precision */
#define PLOSS           6       /* partial loss of precision */
#define TOOMANY         7       /* too many iterations */

typedef void *scf_error_handler_t(char *func_name, int code, char *code_name,
                                  char *msg);

void scf_error_set_ignore(int code, int ignore);
void scf_error(char *func_name, int code, char *msg_fmt, ...);
void scf_error_set_handler(scf_error_handler_t *handler);

#define ASSERT(x) assert(x)

/*
 * Functions
 */

/* Evaluating polynomials and rationals */
double scf_evaluate_polynomial(const double *c, double z, int count);
double scf_evaluate_polynomial_rev(const double *c, double z, int count);
double scf_evaluate_rational(const double *num, const double *denom,
                             double z, int count);

float scf_evaluate_polynomialf(const float *c, float z, int count);
float scf_evaluate_polynomial_revf(const float *c, float z, int count);
float scf_evaluate_rationalf(const float *num, const float *denom, float z,
                             int count);

npy_longdouble scf_evaluate_polynomiall(const npy_longdouble *c,
                                        npy_longdouble z, int count);
npy_longdouble scf_evaluate_polynomial_revl(const npy_longdouble *c,
                                            npy_longdouble z, int count);
npy_longdouble scf_evaluate_rationall(const npy_longdouble *num,
                                      const npy_longdouble *denom,
                                      npy_longdouble z, int count);

/* Bessel I, real-valued */
double scf_iv(double v, double x);
float scf_ivf(float v, float x);
npy_longdouble scf_ivl(npy_longdouble v, npy_longdouble x);

/* Gamma function */
double scf_gamma(double x);
float scf_gammaf(float x);
npy_longdouble scf_gammal(npy_longdouble x);


#endif /* __SCIPYFUNC_H_ */
