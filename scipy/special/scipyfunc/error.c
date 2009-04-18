#include <stdio.h>

#include "scipyfunc.h"

void scf_error(char *name, int code)
{
    printf("%s: %d\n", name, code);
}

