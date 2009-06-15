#ifndef __SCIPYFUNC_H_
#define __SCIPYFUNC_H_

#include <Python.h>
#include <numpy/npy_math.h>
#include <math.h>
#include <stdlib.h>

#include "c99_compat.h"

/*
 * Floating-point precision constants
 */

extern float scf_epsf;
extern float scf_maxnumf;
extern float scf_maxlogf;

extern double scf_eps;
extern double scf_maxnum;
extern double scf_maxlog;

extern npy_longdouble scf_epsl;
extern npy_longdouble scf_maxnuml;
extern npy_longdouble scf_maxlogl;

void scf_init();

/*
 * Mathematical constants
 */

#define EULER  0.577215664901532860606512090082402
#define EULERF 0.577215664901532860606512090082402F
#define EULERL 0.577215664901532860606512090082402L

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
