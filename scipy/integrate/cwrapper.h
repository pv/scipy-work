/* This header file contains wrappers for each of the double quad routines 
  in the SLATEC quadpack library.  Global variables are used to store parameters
  for functions of multiple variables which can then be evaluated through the 
  call function.  The intent is then to wrap this with python and allow use 
  in the SciPy library 
  Authors: Brian Newsom + Nate Woods
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* globalargs; //Array to store function parameters (x[1],...,x[n])
double (*globalf)(int, double *); //Pointer to function of array
int globalnargs; //Int to store number of elements in globargs

int init(double (*f)(int, double *), int n, double args[n]){
  /*Initialize function of n+1 variables
  Input: 
    f - Function pointer to function to evaluate
    n - integer number of extra parameters 
    args - double array of length n with parameters x[1]....x[n]
  Output:
    1 on failure 
    0 on success
  */
  printf("BREAK1\n");
  globalnargs = n;
  printf("BREAK2\n");
  globalf = f;
  printf("BREAK3\n"); 
  globalargs = args;
  /* Apparently this doesn't actually do anything
  if (&globalnargs == NULL || &globalf == NULL || &globalnargs == NULL){
    printf("%s\n", "Initialization did not complete correctly.");
    return 1;
  }*/
  return 0;
}

double call(double* x){ 
  /*Evaluates user defined function as function of one variable. 
    MUST BE INITIALIZED FIRST
  Input: Pointer to double x to evaluate function at
  Output: Function evaluated at x with initialized parameters
  We want to create a new array with [x0, concatenated with [x1, . . . , xn]]
  */ 

  double evalArray[globalnargs+1];
  int i = 1;
  evalArray[0] = *x;

  for(i; i < globalnargs + 1 ; i++){
    evalArray[i] = globalargs[i-1]; //Add everything from globalargs to end of evalArray
  }
  
  return  globalf(globalnargs, evalArray);
}

double dqag2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b,
       double* epsabs, double* epsrel, int* key, double* result, double* abserr, int* neval, int* ier,
       int* limit, int* lenw, int* last, int iwork[*limit], double work[*lenw]){
  init(f,nargs,args);
  dqag_(call,a,b,epsabs,epsrel,key,result,abserr,neval,ier,limit,lenw,last,iwork,work);
  return *result;
}

double dqage2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, double* epsabs,
	      double* epsrel, int* key, int* limit, double* result, double* abserr, int* neval, int* ier, 
	      double alist[*limit], double blist[*limit], double rlist[*limit], double elist[*limit], 
        int iord[*limit], int* last){
  init(f,nargs,args);
  dqage_(call, a,b,epsabs,epsrel,key,limit,result,abserr,neval,ier,alist,blist,rlist,elist,iord,last);
  return *result;
}

double dqagi2(double (*f)(int, double *), int nargs, double args[nargs], double* bound, int* inf,
	      double* epsabs, double* epsrel, double* result, double* abserr, int* neval, int* ier,
	      int* limit, int* lenw, int* last, int iwork[*limit], double work[*lenw]){
  init(f,nargs,args);
  dqagi_(call,bound,inf,epsabs,epsrel,result,abserr,neval,ier,limit,lenw,last,iwork,work);
  return *result;
}

double dqagie2(double (*f)(int, double *), int nargs, double args[nargs], double* bound, int* inf, 
	      double* epsabs, double* epsrel, int* limit, double* result, double* abserr, int* neval, 
	      int* ier, double alist[*limit], double blist[*limit], double rlist[*limit], double elist[*limit], 
	      int iord[*limit], int* last){
  init(f,nargs,args);
  //Weird bug lives here
  // int i = 0;
  // double param = 0;
  // for (i; i < 100 ; i++){
  //   param = i -10e-10;
  //   float paramint = i -10e-10;
  //   printf("%.20f %.20f\n", call(&param), HARDCODEDFUNCTION(&paramint) );
  // }

  //double param = .1;
  //printf("%f Test \n",call(&param));
  //This returns okay...
  //If I skip over my function wrapper and call
  //dqagie_(HARDCODEDFUNCTION, bound, inf, epsabs, epsrel, 
  //  limit, result, abserr, neval, ier, alist, blist, rlist, elist, iord, last);
  //directly, it works.
  //But if I use my wrapper (which shouldn't do anything...)
  dqagie_(call, bound, inf, epsabs, epsrel, limit, result, abserr, neval, ier, alist, blist, 
          rlist, elist, iord, last);
  return *result;

}

double dqags2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b,
	      double* epsabs, double* epsrel, double* result, double* abserr, int* neval, int* ier,
	      int* limit, int* lenw, int* last, int iwork[*limit], double work[*lenw]){//User called function
  init(f,nargs,args);
  dqags_(call,a,b,epsabs,epsrel,result,abserr,neval,ier,limit,lenw,last,iwork,work);
  return *result;
}

void dqagse2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
	      double* epsabs, double* epsrel, int* limit, double* result, double* abserr, int* neval, 
	      int* ier, double alist[*limit], double blist[*limit], double rlist[*limit], 
	      double elist[*limit], int iord[*limit], int* last){
  init(f,nargs,args);
  dqagse_(call, a, b, epsabs, epsrel, limit, result, abserr, neval, ier, alist, blist, rlist, 
	  elist, iord, last);
  return;
}

double dqng2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
        double* epsabs, double* epsrel, double* result, double* abserr, int* neval, int* ier){
  init(f,nargs,args);
  dqng_(call,a,b,epsabs,epsrel,result,abserr,neval,ier);
  return *result;
}

double dqawo2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b,
	      double* omega, int* integr, double* epsabs, double* epsrel, double* result, double* abserr,
	      int* neval, int* ier, int* leniw, int* maxp1, int* lenw, int* last, int iwork[*leniw],
	      double work[*lenw]){
  init(f,nargs,args);
  dqawo_(call, a, b, omega, integr, epsabs, epsrel, result, abserr, neval, ier, 
	 leniw, maxp1, lenw, last, iwork, work);
  return *result;
}


double dqawoe2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
	      double* omega, int* integr, double* epsabs, double* epsrel, int* limit, int* icall, 
	      int* maxp1, double* result, double* abserr, int* neval, int* ier, int* last, 
        double alist[*limit], double blist[*limit], double rlist[*limit], double elist[*limit], 
        int iord[*limit], int nnlog[*limit], int* momcom, double chebmo[*maxp1][25] ){  
  
  init(f,nargs,args);
  dqawoe_(call, a, b, omega, integr, epsabs, epsrel, limit, icall, maxp1, result, abserr, 
        neval, ier, last, alist, blist, rlist, elist, iord, nnlog, momcom, chebmo);
  return *result;
}

double dqagp2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
	      int* npts2, double points[*npts2], double* epsabs, double* epsrel, double* result,
	      double* abserr, int* neval, int* ier, int* leniw, int* lenw, int* last, 
	      int iwork[*leniw], double work[*lenw]){
  init(f,nargs,args);
  dqagp_(call,a,b,npts2,points,epsabs,epsrel,result,abserr,neval,ier,leniw,lenw,
        last,iwork,work);
  return *result;
}

double dqagpe2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
	      int* npts2, double points[*npts2], double* epsabs, double* epsrel, int* limit,
	      double* result, double* abserr, int* neval, int* ier, double alist[*limit], 
	      double blist[*limit], double rlist[*limit], double elist[*limit], double pts[*npts2], 
	      int iord[*limit], int level[*limit], int ndin[*limit], int* last){
  init(f,nargs,args);
  dqagpe_(call, a, b, npts2, points, epsabs, epsrel, limit, result, abserr, neval, ier, alist, 
	      blist, rlist, elist, pts, iord, level, ndin, last);
  return *result;
}

double dqawc2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b,
	      double* c, double* epsabs, double* epsrel, double* result, double* abserr,
	      int* neval, int* ier, int* limit, int* lenw, int* last, int iwork[*limit], 
        double work[*lenw]){
  init(f,nargs,args);
  dqawc_(call,a,b,c,epsabs,epsrel,result,abserr,neval,ier,limit,lenw,last,iwork,work);
  return *result; 
}

double dqawce2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
        double* c, double* epsabs, double* epsrel, int* limit, double* result, double* abserr, 
        int* neval, int* ier, double alist[*limit], double blist[*limit],double rlist[*limit], 
        double elist[*limit], int iord[*limit], int* last){
  init(f,nargs,args);
  dqawce_(call, a, b, c, epsabs, epsrel, limit, result, abserr, neval, ier, alist, blist, rlist, 
        elist, iord, last);
  return *result;
}

double dqawf2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* omega,
	      int* integr, double* epsabs, double* result, double* abserr, int* neval, int* ier, 
	      int* limlst, int* lst, int* leniw, int* maxp1, int* lenw, int iwork[*leniw], 
	      double work[*lenw]){
  init(f,nargs,args);
  dqawf_(call,a,omega,integr,epsabs,result,abserr,neval,ier,limlst,lst,leniw,maxp1,
	      lenw,iwork,work);
  return *result;
}

double dqawfe2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* omega, 
          int* integr, double* epsabs, int* limlst, int* limit, int* maxp1, double* result, 
	        double* abserr, int* neval, int* ier, double rslst[*limlst], double erlst[*limlst], 
	        int ierlst[*limlst], int* lst, double alist[*limit], double blist[*limit], double rlist[*limit], 
	        double elist[*limit], int iord[*limit], int nnlog[*limit], double chebmo[*maxp1][25] ){
  init(f,nargs,args);
  dqawfe_(call, a, omega, integr, epsabs, limlst, limit, maxp1, result, abserr, neval, ier, rslst, 
          erlst, ierlst, lst, alist, blist, rlist, elist, iord, nnlog, chebmo);
  return *result;
}

double dqaws2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b,
	      double* alfa, double* beta, int* integr, double* epsabs, double* epsrel, double* result,
	      double* abserr, int* neval, int* ier, int* limit, int* lenw, int* last, int iwork[*limit],
	      double work[*lenw]){
  init(f,nargs,args);
  dqaws_(call, a, b, alfa, beta, integr, epsabs, epsrel,
	      result, abserr, neval, ier, limit, lenw, last, iwork, work);
  return *result;
}

double dqawse2(double (*f)(int, double *), int nargs, double args[nargs], double* a, double* b, 
 	      double* alfa, double* beta, int* integr, double* epsabs, double* epsrel, int* limit, 
	      double* result, double* abserr, int* neval, int* ier, double alist[*limit], double blist[*limit], 
	      double rlist[*limit], double elist[*limit], int iord[*limit], int* last){
  init(f,nargs,args);
  dqawse_(call, a, b, alfa, beta, integr, epsabs, epsrel, limit, result, abserr, neval, 
        ier, alist, blist, rlist,elist, iord, last);
  return *result;
}
