#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

void mmfree(int n, ...)
{
	va_list ap;
	va_start(ap, n);
	
	for (int i = 0; i < n; ++i) {
		void *p = va_arg(ap, void *);
		if (p) {
			free(p);
			p = NULL;
		}
	}
	
	va_end(ap);
}

void mset(char *const X, size_t size, const char *const val, int nvals)
{
	for (int i = 0; i < nvals; ++i) {
		for (size_t j = 0; j < size; j += nvals) {
			X[j + i] = val[i];
		}
	}
}

void mcopy(const char *const X, char *const Y, size_t size)
{
	for (size_t i = 0; i < size; ++i) {
		Y[i] = X[i];
	}
}