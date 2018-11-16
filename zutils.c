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

void mset(void *X, int size, const char *type, void *value)
{
	if (!X || size <= 0) {
		fprintf(stderr, "X is null[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	if (!strcmp(type, "float")) {
		float *at = (float *)X;
		float alpha = *((float *)value);
		for (int i = 0; i < size; ++i) {
			at[i] = alpha;
		}
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}