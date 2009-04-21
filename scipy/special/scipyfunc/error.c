#include <stdio.h>
#include <stdarg.h>

#include "scipyfunc.h"

static scf_error_handler_t *error_handler = NULL;
static int error_ignore_mask[8] = {
    1, /* DOMAIN */
    1, /* SING */
    1, /* OVERFLOW */
    1, /* UNDERFLOW */
    0, /* TLOSS */
    0, /* PLOSS */
    0  /* TOOMANY */
};

/**
 * Set custom error/warning handler.
 */
void scf_error_set_handler(scf_error_handler_t *handler)
{
    error_handler = handler;
}

/**
 * Set error/warning ignoring status.
 */
void scf_error_set_ignore(int code, int ignore)
{
    if (code >= 0 && code < 8) {
        error_ignore_mask[code] = ignore;
    }
}

/**
 * Issue an error/warning message.
 */
void scf_error(char *name, int code, char *msg_fmt, ...)
{
    char *error_name;
    char fmt_buf[1024];
    va_list ap;

    if (code >= 0 && code < 8 && error_ignore_mask[code]) {
        return;
    }
    
    switch (code) {
    case DOMAIN:
        error_name = "domain";
        break;
    case SING:
        error_name = "singularity";
        break;
    case OVERFLOW:
        error_name = "overflow";
        break;
    case UNDERFLOW:
        error_name = "underflow";
        break;
    case TLOSS:
        error_name = "total loss of precision";
        break;
    case PLOSS:
        error_name = "partial loss of precision";
        break;
    case TOOMANY:
        error_name = "too many iterations";
        break;
    default:
        error_name = "unknown error";
        break;
    }

    va_start(ap, msg_fmt);
    PyOS_vsnprintf(fmt_buf, 1024, msg_fmt, ap);
    va_end(ap);
    
    if (error_handler) {
        error_handler(name, code, error_name, fmt_buf);
    } else {
        /*printf("%s: %d\n", name, code);*/
    }
}
