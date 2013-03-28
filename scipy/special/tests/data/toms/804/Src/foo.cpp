/*            Standard include libraries     */
#include   <cstdio>
#include   <cstdlib>
#include   <iostream>
#include   <fstream>
#include   <cmath>

/*       Include libraries  part of Mathieu functions Library  */
#include   "mathur.h"
#include   "bsslr.h"
#include   "mmathur.h"
#include   "mbsslr.h"
/*
  The Author would be happy to help in any way he can
  in the use of these routins.

  Contact Address:
  CERI, KACST, P.O. Box 6086,
  Riyadh 11442,
  Saudi Arabia,
  Fax:966+1+4813764
  Email: alhargan@kacst.edu.sa

  Legal Matters:
  The program is here as is, no guarantees are given, stated or implied,
  you may use it at your own risk.

  Last modified 20/5/1999

*/

struct point {
   double x;
   double y;
};
/*
  This a driver for the Mathieu functions library
  The following files must be compiled and linked:
  MATHUR.CPP     Mathieu functions
  MMATHUR.CPP    Modified Mathieu functions
  MCNR.CPP       Mathieu charactristic Numbers
  BSSLR.CPP      Bessel functions
  MBSSLR.CPP     Modified Bessel functions
*/
    double Radial(char typ,int n,double h,double u,int kind, double mdf);
    double Circumf(char typ,int n,double h,double v,int kind, double mdf);

/*----------------- Main Porgram --------------------------*/

double scipy_modsem2(int n, double q, double x);

int  main (void)
{
    double z2, z1, r;
    printf("%.22g\n", scipy_modsem2(2, 100, -1));
    printf("%.22g\n",
           (scipy_modsem2(2, 100, -1 + 1e-7) - scipy_modsem2(2, 100, -1)) / 1e-7);
    exit(0);
}

double scipy_modsem2(int n, double q, double x)
{
    double h;
    h = 2 * sqrt(q);
    return Radial('o', n, h, x, 2, 0) / sqrt(M_PI/2);
}

double Radial(char typ,int n,double h,double u,int kind, double mdf)
{

  if(mdf==1) return MathuMZn(typ,n,h,u,kind);
  return MathuZn(typ,n,h,u,kind);
}

double Circumf(char typ,int n,double h,double v,int kind, double mdf)
{

  if(mdf==1) return MathuQn(typ,n,h,v,kind);
  return MathuSn(typ,n,h,v,kind);
}
