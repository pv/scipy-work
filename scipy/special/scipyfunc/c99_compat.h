#ifndef __SCIPYFUNC_C99_COMPAT
#define __SCIPYFUNC_C99_COMPAT

#include <math.h>

/*
 * Ensure C99-similar math functions are available
 */

#if defined(__unix__) || defined(_unix)
#include <unistd.h>
#endif

#if __STDC_VERSION__ < 199901L && _POSIX_VERSION < 200112L

/* C89: #define sin npy_sin */
#define cos npy_cos
/* C89: #define tan npy_tan */
/* C89: #define sinh npy_sinh */
#define cosh npy_cosh
/* C89: #define tanh npy_tanh */
/* C89: #define fabs npy_fabs */
/* C89: #define floor npy_floor */
/* C89: #define ceil npy_ceil */
#define rint npy_rint
#define trunc npy_trunc
/* C89: #define sqrt npy_sqrt */
/* C89: #define log10 npy_log10 */
/* C89: #define log npy_log */
/* C89: #define exp npy_exp */
#define expm1 npy_expm1
/* C89: #define asin npy_asin */
/* C89: #define acos npy_acos */
/* C89: #define atan npy_atan */
/* C89: #define asinh npy_asinh */
/* C89: #define acosh npy_acosh */
/* C89: #define atanh npy_atanh */
#define log1p npy_log1p
/* C89: #define exp2 npy_exp2 */
/* C89: #define log2 npy_log2 */
/* C89: #define atan2 npy_atan2 */
#define hypot npy_hypot
/* C89: #define pow npy_pow */
/* C89: #define fmod npy_fmod */
/* C89: #define modf npy_modf */

#define sinf npy_sinf
#define cosf npy_cosf
#define tanf npy_tanf
#define sinhf npy_sinhf
#define coshf npy_coshf
#define tanhf npy_tanhf
#define fabsf npy_fabsf
#define floorf npy_floorf
#define ceilf npy_ceilf
#define rintf npy_rintf
#define truncf npy_truncf
#define sqrtf npy_sqrtf
#define log10f npy_log10f
#define logf npy_logf
#define expf npy_expf
#define expm1f npy_expm1f
#define asinf npy_asinf
#define acosf npy_acosf
#define atanf npy_atanf
#define asinhf npy_asinhf
#define acoshf npy_acoshf
#define atanhf npy_atanhf
#define log1pf npy_log1pf
#define exp2f npy_exp2f
#define log2f npy_log2f
#define atan2f npy_atan2f
#define hypotf npy_hypotf
#define powf npy_powf
#define fmodf npy_fmodf
#define modff npy_modff

#define sinl npy_sinl
#define cosl npy_cosl
#define tanl npy_tanl
#define sinhl npy_sinhl
#define coshl npy_coshl
#define tanhl npy_tanhl
#define fabsl npy_fabsl
#define floorl npy_floorl
#define ceill npy_ceill
#define rintl npy_rintl
#define truncl npy_truncl
#define sqrtl npy_sqrtl
#define log10l npy_log10l
#define logl npy_logl
#define expl npy_expl
#define expm1l npy_expm1l
#define asinl npy_asinl
#define acosl npy_acosl
#define atanl npy_atanl
#define asinhl npy_asinhl
#define acoshl npy_acoshl
#define atanhl npy_atanhl
#define log1pl npy_log1pl
#define exp2l npy_exp2l
#define log2l npy_log2l
#define atan2l npy_atan2l
#define hypotl npy_hypotl
#define powl npy_powl
#define fmodl npy_fmodl
#define modfl npy_modfl

#endif

/*
 * Additional macros
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

#ifndef PI
#define PI NPY_PI
#endif
#define PIF NPY_PIf
#define PIL NPY_PIl

#endif /* __SCIPYFUNC_C99_COMPAT */
