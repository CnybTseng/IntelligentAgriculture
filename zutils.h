#ifndef _ZUTILS_H_
#define _ZUTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef OPENCL
#	include "CL/opencl.h"
#endif

#ifdef __linux__
#define BINARY_FILENAME_TO_START(name, suffix) \
	_binary_##name##_##suffix##_start
#define BINARY_FILENAME_TO_END(name, suffix) \
	_binary_##name##_##suffix##_end
#define BINARY_FILENAME_TO_SIZE(name, suffix) \
	_binary_##name##_##suffix##_size
#elif defined(_WIN32)
#define BINARY_FILENAME_TO_START(name, suffix) \
	binary_##name##_##suffix##_start
#define BINARY_FILENAME_TO_END(name, suffix) \
	binary_##name##_##suffix##_end	
#define BINARY_FILENAME_TO_SIZE(name, suffix) \
	binary_##name##_##suffix##_size
#else
#	error "unsupported operation system!"
#endif

#ifdef OPENCL
#ifdef FLOAT
#	define PARSE_PRECISION strcat(options, " -DFLOAT -DDATA_TYPE=float -DREAD_WRITE_DATA_TYPE=f")
#	define IMAGE_CHANNEL_DATA_TYPE CL_FLOAT
#	define MEM_MAP_PTR_TYPE cl_float
#else
#	define PARSE_PRECISION strcat(options, " -DDATA_TYPE=half -DREAD_WRITE_DATA_TYPE=h")
#	define IMAGE_CHANNEL_DATA_TYPE CL_HALF_FLOAT
#	define MEM_MAP_PTR_TYPE cl_half
#endif
#endif

void mmfree(int n, ...);
void mset(char *const X, size_t size, const char *const val, int nvals);
void mcopy(const char *const X, char *const Y, size_t size);
void save_volume(float *data, int width, int height, int nchannels, const char *path);
#ifdef OPENCL
int nchw_to_nhwc(const float *const input, MEM_MAP_PTR_TYPE *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size);
int nhwc_to_nchw(const MEM_MAP_PTR_TYPE *const input, float *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size);
#endif
int round_up_division_2(int x);
int round_up_division_4(int x);
unsigned int roundup_power_of_2(unsigned int a);
unsigned int round_up_multiple_of_8(unsigned int x);

#ifdef __cplusplus
}
#endif

#endif