/*Second wrapper for testing. Interprets fn of f(x) as f(n,x[n]) for use with
cwrapper*/

double (*globalbasef)(double *); //Single variate function

void funcwrapper_init(double (*f)(double *)){
	//sets f as global for future use
	//input: f - function of double pointer
	printf("Inside funcwrapper_init\n");
	globalbasef = f;
	printf("Inside2\n");
	double param = 2.0;
	printf("%f\n",globalbasef(&param));
	return;
}

double funcwrapper(int nargs, double args[nargs]){
	return 12.0;
}