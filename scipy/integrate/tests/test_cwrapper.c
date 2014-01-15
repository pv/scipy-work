/* This c file is meant as proof of concept to show that the fulllib.h
  header can serve the expected purpose and will allow the usage of 
  multivariate functions declared in c to speed up scipy 
  Authors: Brian Newsom + Nate Woods
  */
#include "cwrapper.h"

static const double  PI=3.14159265359;

int assert_quad(double value, double actualValue, double errTol){
  /* Test if quad returns within expected tolerance.
  Input:
    value - Return value through quadpack
    actualValue - Known value of integral
    errTol - Maximum error allowed
    
  Output:
    1 - Error is larger than tolerance
    0 - Error within tolerance
  */
  if (fabs(value - actualValue) > errTol){
    printf("Integrated value: %f\n", value);
    printf("Expected value: %f\n", actualValue);
    printf("Error: %f is greater than tolerance\n", fabs(value-actualValue));
    return 1;
  }
  else{
    printf("Integration within tolerance\n");
    return 0; 
  }
}

double ttFunc(int nargs, double args[nargs]){
  return cos(args[1]*args[0] - args[2]*sin(args[0])) / PI;
}

int test_typical(){
  
  int nargs = 2;
  double args[nargs];
  args[0] = 2;
  args[1] = 1.8;
  double a = 0;
  double b = PI;
  double epsabs = 1.49E-8;
  double epsrel = 1.49E-8;
  int limit = 50;
  double result, abserr;
  int neval, ier;
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  int iord[limit];
  int last;
  double value = dqagse2(ttFunc, nargs, args, &a, &b, &epsabs, &epsrel, &limit, &result, 
        &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);

  return assert_quad(value, 0.30614353532540296487, 1.49e-8);

}

double tiFunc(int nargs, double args[nargs]){
  return -(exp(-args[0]))*log(args[0]);
}

double tiFunc2(double* x){
  return -(exp(-*x)*log(*x));
}

int test_indefinite(){
  int nargs = 0;
  double args[nargs];
  double bound = 0.0;
  int inf = 1;
  double epsabs = 1.49E-9;
  double epsrel = 1.49E-8;
  int limit = 100;
  double result, abserr;
  int neval, ier;
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  int iord[limit];
  int last;
  double value = dqagie2(tiFunc, nargs, args, &bound, &inf, &epsabs, &epsrel, &limit, &result, 
			 &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last); //ERROR
  //double value2 = dqagie_(tiFunc2, &bound, &inf, &epsabs, &epsrel, &limit, &result, &abserr, &neval, 
	//		  &ier, alist, blist, rlist, elist, iord, &last); // If i skip the wrapper it works.
  //value2 = result; //Somehow this one is correct..
  return assert_quad(value, 0.577215664901532860606512, 1.49e-8);

}

double tsFunc(int nargs, double args[nargs]){
  if (args[0] > 0 && args[0] < 2.5)
    return sin(args[0]);
  else if( args[0] >= 2.5 && args[0] <= 5.0)
    return exp(-args[0]);
  else
    return 0.0;
}

int test_singular(){

  int nargs = 0;
  double args[nargs];
  double a = 0;
  double b = 10;
  int npts2 = 4;
  double points[npts2];
  points[0] = 2.5;
  points[1] = 5.0;
  double epsabs = 1.49E-8;
  double epsrel = 1.49E-8;
  int limit = 50;
  double result, abserr;
  int neval, ier;
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  double pts[npts2];
  int iord[limit];
  int level[limit];
  int ndin[limit];
  int last;
  
  double value = dqagpe2(tsFunc, nargs, args, &a, &b, &npts2, points, &epsabs, &epsrel,
        &limit, &result, &abserr, &neval, &ier, alist, blist, rlist,elist, pts, iord,
        level, ndin, &last);
  return assert_quad(value, 1 - cos(2.5) + exp(-2.5) - exp(-5.0), 1.49e-8);
}

double tswfFunc(int nargs, double args[nargs]){
  return exp(args[1]*(args[0]-1));
}

int test_sine_weighted_finite(){

  int nargs = 1;
  double args[nargs];
  args[0] = 20;
  double a = 0;
  double b = 1;
  double omega = pow(2,3.4);
  int integr = 2; 
  double epsabs = 1.49E-8;
  double epsrel = 1.49E-8;
  int limit = 50;
  int icall = 1;
  int maxp1 = 50;

  double result, abserr;
  int neval, ier;
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  int iord[limit];
  int nnlog[limit];
  int momcom = 0;
  double chebmo[maxp1][25]; 
  int last;

  double value = dqawoe2(tswfFunc, nargs, args, &a, &b, &omega, &integr, &epsabs, &epsrel, 
        &limit, &icall, &maxp1, &result, &abserr, &neval, &ier, &last, alist, blist, rlist, 
        elist, iord, nnlog, &momcom, chebmo);
  return assert_quad(value, (20*sin(omega)-omega*cos(omega)+omega*exp(-20))/(pow(20,2) 
        + pow(omega,2)), 1.49e-8);
}

double tswiFunc(int nargs, double args[nargs]){
  return exp(-args[0]*args[1]);
}


int test_sine_weighted_infinite(){

  int nargs = 1;
  double args[nargs];
  args[0] = 4.0;
  double a = 0;
  double omega = 3.0;
  int integr = 2;
  double epsabs = 1.49E-8;
  int limlst = 50;
  int limit = 50;
  int maxp1 = 50;
  
  double result, abserr;
  int neval, ier;
  double rslst[limlst];
  double erlst[limlst];
  int ierlst[limlst];
  int lst = 50; 
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  int iord[limit];
  int nnlog[limit];
  double chebmo[maxp1][25];                                                         

  double value = dqawfe2(tswiFunc, nargs, args, &a, &omega, &integr, &epsabs, &limlst, 
        &limit, &maxp1, &result, &abserr, &neval, &ier, rslst, erlst, ierlst, &lst, alist, 
        blist, rlist, elist, iord, nnlog, chebmo);
  return assert_quad(value, 3.0/(pow(4,2) + pow(3,2)), 1.49e-8);
}

double tcwiFunc(int nargs, double args[nargs]){
  return exp(args[0]*args[1]);
}

int test_cosine_weighted_infinite(){
  printf("This is done in a weird way I don't know how to translate\n");
  printf("It seems to essentially change the function based on symmetry or something\n");
  /*
  int nargs = 1;
  double args[nargs];
  args[0] = 2.5;
  double a = 0;
  double omega = 2.3;
  int integr = 1;
  double epsabs = 1.49E-8;
  int limlst = 50;
  int limit = 50;
  int maxp1 = 50;

  double result, abserr;
  int neval, ier;
  double rslst[limlst];
  double erlst[limlst];
  int ierlst[limlst];
  int lst = 50; //Not sure on this value                                                                                                    
  double alist[limit];
  double blist[limit];
  double rlist[limit];
  double elist[limit];
  int iord[limit];
  int nnlog[limit];
  double chebmo[maxp1][25]; //This 2D array could have problems w/ fortran and c?? 
  double value = dqawfe2(tcwiFunc, nargs, args, &a, &omega, &integr, &epsabs, &limlst,
      &limit, &maxp1, &result, &abserr, &neval, &ier, rslst, erlst, ierlst, &lst, alist, 
      blist, rlist, elist, iord, nnlog, chebmo);
  return assert_quad(value, 2.5/(pow(2.5,2.3) + pow(3,2)), 1.49e-8);
  */
}

double talwFunc(int nargs, double args[nargs]){
  return 1/(1+args[0]+pow(2,(-args[1])));
}

int test_algebraic_log_weight(){
  int nargs = 1;
  double args[nargs];
  args[0] = 1.5;
  double a = -1;
  double b = 1;
  double alfa = -.5;
  double beta = -.5; 
  int integr = 1; //corresponds to strdict[1]
  double epsabs = 1.49e-8;
  double epsrel = 1.49e-8;
  int limit = 50;
  double result, abserr;
  int neval, ier;
  double alist[limit], blist[limit], rlist[limit], elist[limit];
  int iord[limit];
  int last;
  double value = dqawse2(talwFunc, nargs, args, &a, &b, &alfa, &beta, &integr, &epsabs, 
        &epsrel, &limit, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, 
        iord, &last);
  return assert_quad(value,  PI/sqrt(pow((1+pow(2,(-1.5))),2) - 1), 1.49e-8);
} 

double tcwFunc(int nargs, double args[nargs]){
  return pow(2.0,(-args[1]))/(pow((args[0]-1),2)+pow(4.0,(-args[1])));
}

int test_cauchypv_weight(){
  int nargs = 1;
  double args[nargs];
  args[0] = .4;
  double a = 0;
  double b = 5;
  double c = 2.0; //wvar
  double epsabs = 1.49e-8;
  double epsrel = 1.49e-8;
  int limit = 50;
  double result, abserr;
  int neval, ier;
  double alist[limit], blist[limit], rlist[limit], elist[limit];
  int iord[limit];
  int last;
  double value = dqawce2(tcwFunc, nargs, args, &a, &b, &c, &epsabs, &epsrel, &limit, 
        &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
  a = 0.4;
  double tabledValue = (pow(2.0,(-0.4))*log(1.5)-pow(2.0,(-1.4))*
        log((pow(4.0,(-a))+16)/(pow(4.0,(-a))+1)) - atan(pow(2.0,(a+2))) 
        - atan(pow(2.0,a)))/(pow(4.0,(-a)) + 1);
  return assert_quad(value, tabledValue, 1.49e-8);
  
}

int main(){
  printf("Test Typical\n");
  test_typical();
  printf("Test Indefinite\n");
  test_indefinite();
  printf("Test Singlar\n");
  test_singular();
  printf("Test Sine Weighted Finite\n");
  test_sine_weighted_finite();
  printf("Test Sine Weighted Infinite\n");
  test_sine_weighted_infinite();
  printf("Test Cosine Weighted Infinite\n");
  test_cosine_weighted_infinite();
  printf("Test Algebraic Log Weight\n");
  test_algebraic_log_weight();
  printf("Test Cauchypv Weight\n");
  test_cauchypv_weight();
  return 0;

}
