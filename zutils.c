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

void nchw_to_nhwc_quad(const float *const input, float *const output, int width, int height, int channels, int batch)
{
	const int channel_blocks = (channels + 3) >> 2;
	const int input_slice_pitch = width * height;
	const int output_row_pitch = ((width << 2) * channel_blocks) * batch;
	const int batch_size = width * height * channels;
	for (int b = 0; b < batch; ++b) {
		const float *src_batch = input + b * batch_size;
		float *dst_batch = output + b * ((width << 2) * channel_blocks);
		for (int k = 0; k < channel_blocks; ++k) {
			const float *src = src_batch + k * (input_slice_pitch << 2);
			float *dst = dst_batch + k * (width << 2);
			int channel_remainder = channels - (k << 2);
			channel_remainder = channel_remainder < 4 ? channel_remainder : 4;
			for (int c = 0; c < channel_remainder; ++c) {
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						dst[y * output_row_pitch + (x << 2) + c] = src[c * input_slice_pitch + y * width + x];
					}
				}
			}
			for (int c = channel_remainder; c < 4; ++c) {
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						dst[y * output_row_pitch + (x << 2) + c] = 0;
					}
				}
			}
		}
	}
}