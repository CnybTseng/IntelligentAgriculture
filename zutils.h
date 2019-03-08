#ifndef _ZUTILS_H_
#define _ZUTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

void mmfree(int n, ...);
void mset(char *const X, size_t size, const char *const val, int nvals);
void mcopy(const char *const X, char *const Y, size_t size);
void save_volume(float *data, int width, int height, int nchannels, const char *path);
void nchw_to_nhwc_quad(const float *const input, float *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch);
void nhwc_to_nchw_quad(const float *const input, float *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch);
int round_up_division_2(int x);
int round_up_division_4(int x);

#ifdef __cplusplus
}
#endif

#endif