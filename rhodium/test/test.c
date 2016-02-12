#include <stdio.h>

double norm_return(double x, double y) {
	return x * y;
}

void arg_return(double x, double y, double* z) {
	*z = x * y;
}

double sum(double* x) {
	int i;
	double sum = 0.0;

	for (i = 0; i < 10; i++) {
		sum += x[i];
	}

	return sum;
}

void array_add(double* x, double* y, long n, double* z) {
	int i;
	printf("%d\n", n);

	for (i = 0; i < n; i++) {
		z[i] = x[i] + y[i];
	}
}