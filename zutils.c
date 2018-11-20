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

void save_volume(float *data, int width, int height, int nchannels, const char *path)
{
	FILE *fp = fopen(path, "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	for (int c = 0; c < nchannels; ++c) {
		fprintf(fp, "channel=%d\n", c);
		float *at = data + c * width * height;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				fprintf(fp, "%.7f ", at[y * width + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n", fp);
	}

	fclose(fp);
}