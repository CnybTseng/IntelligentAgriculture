#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "winograd_convolution.h"
#include "gemm.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

#define MAX_FILTER_CHANNELS	1024

float G_f6x6_3x3[32] = {
     1,       0,       0, 0,
-2/9.0,  -2/9.0,  -2/9.0, 0,
-2/9.0,   2/9.0,  -2/9.0, 0,
1/90.0,  1/45.0,  2/45.0, 0,
1/90.0, -1/45.0,  2/45.0, 0,
1/45.0,  1/90.0, 1/180.0, 0,
1/45.0, -1/90.0, 1/180.0, 0,
     0,       0,       1, 0	
};

float BT_f6x6_3x3[64] = {
1,      0, -21/4.0,       0,  21/4.0,       0, -1, 0,
0,      1,       1, -17/4.0, -17/4.0,       1,  1, 0,
0,     -1,       1,  17/4.0, -17/4.0,      -1,  1, 0,
0,  1/2.0,   1/4.0,  -5/2.0,  -5/4.0,       2,  1, 0,
0, -1/2.0,   1/4.0,   5/2.0,  -5/4.0,      -2,  1, 0,
0,      2,       4,  -5/2.0,      -5,   1/2.0,  1, 0,
0,     -2,       4,   5/2.0,      -5,  -1/2.0,  1, 0,
0,     -1,       0,  21/4.0,       0, -21/4.0,  0, 1
};

float AT_f6x6_3x3[64] = {
1, 1,  1,  1,   1, 32,  32, 0,
0, 1, -1,  2,  -2, 16, -16, 0,
0, 1,  1,  4,   4,  8,   8, 0,
0, 1, -1,  8,  -8,  4,  -4, 0,
0, 1,  1, 16,  16,  2,   2, 0,
0, 1, -1, 32, -32,  1,  -1, 1,
0, 0,  0,  0,   0,  0,   0, 0,
0, 0,  0,  0,   0,  0,   0, 0
};

float G_f4x4_3x3[18] = {
     1,                  0,      0,
-2/3.0, -0.471404520791032, -1/3.0,
-2/3.0,  0.471404520791032, -1/3.0,
 1/6.0,  0.235702260395516,  1/3.0,
 1/6.0, -0.235702260395516,  1/3.0,
     0,                  0,      1
};

float BT_f4x4_3x3[36] = {
1,                  0, -5/2.0,                  0, 1, 0,
0, -1.414213562373095,     -2,  0.707106781186548, 1, 0,
0,  1.414213562373095,     -2, -0.707106781186548, 1, 0,
0, -0.707106781186548, -1/2.0,  1.414213562373095, 1, 0,
0,  0.707106781186548, -1/2.0, -1.414213562373095, 1, 0,
0,                  1,      0,             -5/2.0, 0, 1
};

float AT_f4x4_3x3[24] = {
1,                 1,                  1,                   1,                    1, 0,
0, 0.707106781186548, -0.707106781186548,   1.414213562373095,   -1.414213562373095, 0,
0,               1/2,                1/2,                   2,                    2, 0,
0, 0.353553390593274, -0.353553390593274, 2*1.414213562373095, -2*1.414213562373095, 1
};

#ifdef OPENCL
extern cl_wrapper wrapper;

struct weight_transform_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_G;
	cl_mem d_weight;
	cl_mem d_transformed_weight;
	int filter_size;
	int filter_channels;
	int nfilters;
	int transf_weight_size;
	int transformed_weight_width;
	int transformed_weight_height;
	int transformed_weight_depth;
};

struct input_transform_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_BT;
	cl_mem d_input;
	cl_mem d_transformed_input;
	int input_width;
	int input_height;
	int input_channels;
	int stride;
	int padding;
	int end_of_line;
	int ntilesX;
	int ntilesY;
	int transformed_input_image_width;
	int transformed_input_image_height;
	int transformed_input_image_depth;
};

struct matrix_multiplication_context {
	cl_program program;
	cl_kernel kernel;
	weight_transform_context *wtc;
	input_transform_context *itc;
	cl_mem d_output;
};

struct output_inverse_transform_context {
	cl_program program;
	cl_kernel kernel;
	matrix_multiplication_context *mmc;
	cl_mem d_AT;
	cl_mem d_inverse_transformed_output;
	int tile_output_size;
};
#endif

int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv)
{
	int filter_size;
	int conv_tile_output_size;
	if (conv == F6x6_3x3) {
		filter_size = 3;
		conv_tile_output_size = 6;
	} else if (conv == F4x4_3x3) {
		filter_size = 3;
		conv_tile_output_size = 4;
	} else if (conv == F2x2_3x3) {
		filter_size = 3;
		conv_tile_output_size = 2;
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	return filter_size + conv_tile_output_size - 1;
}

int get_convolution_tile_output_size(WINOGRAD_CONV_TYPE conv)
{
	if (conv == F6x6_3x3) {
		return 6;
	} else if (conv == F4x4_3x3) {
		return 4;
	} else if (conv == F2x2_3x3) {
		return 2;
	}
	
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	return 0;
}

#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv, int filter_channels,
	int nfilters)
{
	weight_transform_context *context = calloc(1, sizeof(weight_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->filter_size = 3;
	context->filter_channels = filter_channels;
	context->nfilters = nfilters;
	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "weight_transform_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->d_G = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		8 * 4 * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	float *h_G = clEnqueueMapBuffer(wrapper.command_queue, context->d_G, CL_TRUE, CL_MAP_WRITE,
		0, 8 * 4 * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_G, G_f6x6_3x3, 8 * 4 * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_G, h_G, 0, NULL, NULL);
	
	context->d_weight = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		3 * 3 * MAX_FILTER_CHANNELS * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->transf_weight_size = get_transformed_weight_matrix_size(conv);
	context->transformed_weight_width = ((filter_channels + 3) / 4) * 4;
	context->transformed_weight_height = ((nfilters + 3) / 4) * 4;
	context->transformed_weight_depth = context->transf_weight_size * context->transf_weight_size;
	context->d_transformed_weight = clCreateBuffer(wrapper.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		context->transformed_weight_width * context->transformed_weight_height * context->transformed_weight_depth *
		sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_weight_transform_context(context);
		return 0;
	}
	
	return context;
}

void transform_weight(weight_transform_context *context, float *weights, float *transformed_weights)
{
	cl_int errcode;
	const int standard_batch_size = MAX_FILTER_CHANNELS;
	int num_unprocessed_channels = context->filter_channels * context->nfilters;
	const int num_batches = (num_unprocessed_channels + 1023) >> 10;
#ifdef CL_PROFILING_ENABLE	
	float duration = 0;
#endif
	const int round_filter_channels = ((context->filter_channels + 3) / 4) * 4;
	for (int b = 0; b < num_batches; ++b) {
		int batch_size = num_unprocessed_channels < standard_batch_size ? num_unprocessed_channels : standard_batch_size;
		
		float *h_weight = clEnqueueMapBuffer(wrapper.command_queue, context->d_weight, CL_TRUE, CL_MAP_WRITE,
			0, context->filter_size * context->filter_size * batch_size * sizeof(float), 0, NULL, NULL, &errcode);
		memcpy(h_weight, weights + b * standard_batch_size * context->filter_size * context->filter_size,
			context->filter_size * context->filter_size * batch_size * sizeof(float));
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_weight, h_weight, 0, NULL, NULL);

		const int xy_shift = ((b * standard_batch_size) / context->filter_channels) * round_filter_channels +
			(b * standard_batch_size) % context->filter_channels;
		errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_weight);
		errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_G); 
		errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_transformed_weight); 
		errcode |= clSetKernelArg(context->kernel, 3, sizeof(cl_int), &context->filter_channels);
		errcode |= clSetKernelArg(context->kernel, 4, sizeof(cl_int), &context->nfilters);
		errcode |= clSetKernelArg(context->kernel, 5, sizeof(cl_int), &xy_shift);
		
		cl_event event;
		cl_uint work_dim = 1;
		size_t global_work_size[] = {batch_size, 1, 1};
		clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
			NULL, 0, NULL, &event);
#ifdef CL_PROFILING_ENABLE	
		cl_ulong start, end;
		clFinish(wrapper.command_queue);
		errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		duration += (end - start) * 1e-6f;
#endif
		clReleaseEvent(event);
		num_unprocessed_channels -= batch_size;
	}
#ifdef CL_PROFILING_ENABLE
	printf("weight_transform_f6x6_3x3:%fms(batches %d)\n", duration, num_batches);
#endif

	if (transformed_weights) {
		const int len = context->transformed_weight_width * context->transformed_weight_height *
			context->transformed_weight_depth * sizeof(float);
		float *h_transformed_weight = clEnqueueMapBuffer(wrapper.command_queue, context->d_transformed_weight,
			CL_TRUE, CL_MAP_READ, 0, len, 0, NULL, NULL, &errcode);
		memcpy(transformed_weights, h_transformed_weight, len);
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_weight, h_transformed_weight, 0, NULL, NULL);
	}
}

void free_weight_transform_context(weight_transform_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_G);
		clReleaseMemObject(context->d_weight);
		clReleaseMemObject(context->d_transformed_weight);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

input_transform_context *create_input_transform_context(WINOGRAD_CONV_TYPE conv, int input_width,
	int input_height, int input_channels, int stride, int padding)
{
	input_transform_context *context = calloc(1, sizeof(input_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->input_width = input_width;
	context->input_height = input_height;
	context->input_channels = input_channels;
	context->stride = stride;
	context->padding = padding;
	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "input_transform_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	const int input_size = get_transformed_weight_matrix_size(conv);	
	context->d_BT = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		input_size * input_size * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	float *h_BT = clEnqueueMapBuffer(wrapper.command_queue, context->d_BT, CL_TRUE, CL_MAP_WRITE,
		0, input_size * input_size * sizeof(float), 0, NULL, NULL, &errcode);
	
	if (conv == F6x6_3x3) {
		memcpy(h_BT, BT_f6x6_3x3, input_size * input_size * sizeof(float));
	} else if (conv == F4x4_3x3) {
		memcpy(h_BT, BT_f4x4_3x3, input_size * input_size * sizeof(float));
	} else {
		fprintf(stderr, "Not implemented![%s:%d].\n", __FILE__, __LINE__);
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_BT, h_BT, 0, NULL, NULL);
		goto cleanup;
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_BT, h_BT, 0, NULL, NULL);
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format input_image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};
	
	const int output_size = get_convolution_tile_output_size(conv);
	context->ntilesX = (input_width + (output_size - 1)) / output_size;
	context->ntilesY = (input_height + (output_size - 1)) / output_size;
	cl_image_desc input_image_desc;
	memset(&input_image_desc, 0, sizeof(cl_image_desc));
	input_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	input_image_desc.image_width = (context->ntilesX * input_size) >> 2;
	input_image_desc.image_height = (context->ntilesY * output_size + 2) * input_channels;
	input_image_desc.image_row_pitch = 0;
		
	// printf("alloc input image %dx%dx%d => %d(x4)x%d\n", input_width, input_height, input_channels, input_image_desc.image_width,
	// 	input_image_desc.image_height);
	context->d_input = clCreateImage(wrapper.context, mem_flags, &input_image_format, &input_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	for (int x = 1, sx = 0; x < (input_image_desc.image_width << 2); ++sx) {
		if (sx >= input_width) {
			context->end_of_line = x;
			break;
		}
		sx -= ((++x % 8 == 0) << 1);
	}

	context->transformed_input_image_width = ((context->ntilesX * context->ntilesY + 3) / 4) * 4;
	context->transformed_input_image_height = ((input_channels + 3) / 4) * 4;
	context->transformed_input_image_depth = input_size * input_size;
	context->d_transformed_input = clCreateBuffer(wrapper.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		context->transformed_input_image_width * context->transformed_input_image_height *
		context->transformed_input_image_depth * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_input_transform_context(context);
		return 0;
	}
	
	return context;
}	

void get_transformed_input_size(input_transform_context *context, int *width, int *height, int *array_size)
{
	if (context) {
		*width = context->transformed_input_image_width;
		*height = context->transformed_input_image_height;
		*array_size = context->transformed_input_image_depth;
	} else {
		*width = 0;
		*height = 0;
		*array_size = 0;
	}
}
	
void transform_input(input_transform_context *context, float *input, float *transformed_input)
{
	size_t input_image_origin[] = {0, 0, 0};
	size_t input_image_region[] = {0, 0, 1};
	clGetImageInfo(context->d_input, CL_IMAGE_WIDTH, sizeof(size_t), &input_image_region[0], NULL);
	clGetImageInfo(context->d_input, CL_IMAGE_HEIGHT, sizeof(size_t), &input_image_region[1], NULL);
	
	cl_int errcode;
	size_t input_image_row_pitch, input_image_slice_pitch;
	float *h_input = clEnqueueMapImage(wrapper.command_queue, context->d_input, CL_TRUE, CL_MAP_WRITE, input_image_origin,
		input_image_region, &input_image_row_pitch, &input_image_slice_pitch, 0, NULL, NULL, &errcode);

	input_image_row_pitch = input_image_row_pitch >> 2;
	const int height_per_channel = input_image_region[1] / context->input_channels;
	for (int z = 0; z < context->input_channels; ++z) {
		float *dst = h_input + z * height_per_channel * input_image_row_pitch;
		for (int x = 0; x < input_image_region[0] * 4; ++x) {
			dst[x] = 0;
		}
		float *src = input + z * context->input_width * context->input_height;
		for (int y = 1; y < context->input_height + 1; ++y) {
			dst[y * input_image_row_pitch] = 0;
			int sy = y - 1;
			for (int x = 1, sx = 0; x < context->end_of_line; ++sx) {
				dst[y * input_image_row_pitch + x] = src[sy * context->input_width + sx];
				sx -= ((++x % 8 == 0) << 1);
			}		
			for (int x = context->end_of_line; x < input_image_region[0] * 4; ++x) {
				dst[y * input_image_row_pitch + x] = 0;
			}
		}
		for (int y = context->input_height + 1; y < height_per_channel; ++y) {
			for (int x = 0; x < input_image_region[0] * 4; ++x) {
				dst[y * input_image_row_pitch + x] = 0;
			}
		}
	}
	
	if (transformed_input) {
		save_volume(h_input, input_image_region[0] * 4, height_per_channel, context->input_channels, "formated_inputs.txt");
	}
	
	cl_event event;
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_input, h_input, 0, NULL, &event);
	
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_input);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(int), &height_per_channel);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_BT);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(cl_mem), &context->d_transformed_input);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d].\n", __FILE__, __LINE__);
	}
	printf("ntilesX %d, ntilesY %d\n", context->ntilesX, context->ntilesY);
	cl_uint work_dim = 3;
	size_t global_work_size[] = {context->ntilesX, context->ntilesY, context->input_channels};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	printf("input_transform_f6x6_3x3: %fms\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);

	if (transformed_input) {
		const int len = context->transformed_input_image_width * context->transformed_input_image_height *
			context->transformed_input_image_depth * sizeof(float);
		float *h_transformed_input = clEnqueueMapBuffer(wrapper.command_queue, context->d_transformed_input,
			CL_TRUE, CL_MAP_WRITE, 0, len, 0, NULL, NULL, &errcode);
		memcpy(transformed_input, h_transformed_input, len);	
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_input, h_transformed_input, 0, NULL, NULL);
	}
}

void free_input_transform_context(input_transform_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_BT);
		clReleaseMemObject(context->d_input);
		clReleaseMemObject(context->d_transformed_input);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

matrix_multiplication_context *create_matrix_multiplication_context(WINOGRAD_CONV_TYPE conv,
	weight_transform_context *wtc, input_transform_context *itc)
{
	matrix_multiplication_context *context = calloc(1, sizeof(matrix_multiplication_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->wtc = wtc;
	context->itc = itc;
	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "matrix_multiply", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	const int tile_input_size = get_transformed_weight_matrix_size(conv);
	context->d_output = clCreateBuffer(wrapper.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		wtc->transformed_weight_height * itc->transformed_input_image_width * tile_input_size * tile_input_size *
		sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_matrix_multiplication_context(context);
		return 0;
	}
	
	return context;
}

void muliply_transformed_matrix(matrix_multiplication_context *context, float *output)
{
	cl_int errcode;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->wtc->d_transformed_weight);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->itc->d_transformed_input);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_output);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &context->wtc->transformed_weight_width);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &context->wtc->transformed_weight_height);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &context->wtc->transf_weight_size);
	errcode |= clSetKernelArg(context->kernel, 6, sizeof(int), &context->itc->transformed_input_image_width);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	
	cl_event event;
	cl_uint work_dim = 3;
	size_t global_work_size[] = {context->itc->transformed_input_image_width >> 2,
		context->wtc->transformed_weight_height >> 2, context->wtc->transf_weight_size * context->wtc->transf_weight_size};
	printf("global_work_size %d %d %d\n", global_work_size[0], global_work_size[1], global_work_size[2]);
	printf("matrix size: (%dx%d)x(%dx%d)\n", context->wtc->transformed_weight_height, context->wtc->transformed_weight_width,
		context->wtc->transformed_weight_width, context->itc->transformed_input_image_width);
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	printf("matrix_multiply: %fms\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);

	if (output) {
		const int len = context->wtc->transformed_weight_height * context->itc->transformed_input_image_width *
			context->wtc->transf_weight_size * context->wtc->transf_weight_size * sizeof(float);
		float *h_output = clEnqueueMapBuffer(wrapper.command_queue, context->d_output, CL_TRUE, CL_MAP_WRITE,
			0, len, 0, NULL, NULL, &errcode);
		memcpy(output, h_output, len);	
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_output, h_output, 0, NULL, NULL);
	}
}

void free_matrix_multiplication_context(matrix_multiplication_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

output_inverse_transform_context *create_output_inverse_transform_context(WINOGRAD_CONV_TYPE conv,
	matrix_multiplication_context *mmc)
{
	output_inverse_transform_context *context = calloc(1, sizeof(output_inverse_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->mmc = mmc;
	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "inverse_output_transform_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	const int input_size = get_transformed_weight_matrix_size(conv);	
	context->d_AT = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		input_size * input_size * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	float *h_AT = clEnqueueMapBuffer(wrapper.command_queue, context->d_AT, CL_TRUE, CL_MAP_WRITE,
		0, input_size * input_size * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_AT, AT_f6x6_3x3, input_size * input_size * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_AT, h_AT, 0, NULL, NULL);
	
	context->tile_output_size = get_convolution_tile_output_size(conv);
	context->d_inverse_transformed_output = clCreateBuffer(wrapper.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		 mmc->wtc->nfilters * (mmc->itc->ntilesX * mmc->itc->ntilesY) * (context->tile_output_size *
		 context->tile_output_size) * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_output_inverse_transform_context(context);
		return 0;
	}
	
	return context;
}	

void inverse_transform_output(output_inverse_transform_context *context, float *inverse_transformed_output)
{
	cl_int errcode;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->mmc->d_output);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_inverse_transformed_output);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_AT);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &context->mmc->itc->transformed_input_image_width);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &context->mmc->wtc->transf_weight_size);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &context->mmc->wtc->transformed_weight_height);
	errcode |= clSetKernelArg(context->kernel, 6, sizeof(int), &context->mmc->itc->input_width);
	errcode |= clSetKernelArg(context->kernel, 7, sizeof(int), &context->mmc->itc->input_height);
	errcode |= clSetKernelArg(context->kernel, 8, sizeof(int), &context->mmc->itc->ntilesX);
	errcode |= clSetKernelArg(context->kernel, 9, sizeof(int), &context->mmc->itc->ntilesY);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {context->mmc->itc->ntilesX * context->mmc->itc->ntilesY,
		context->mmc->wtc->nfilters, 1};
	printf("global_work_size %d %d %d\n", global_work_size[0], global_work_size[1], global_work_size[2]);
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	printf("inverse_output_transform_f6x6_3x3: %fms\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);

	if (inverse_transformed_output) {
		const int len = context->mmc->wtc->nfilters * (context->mmc->itc->ntilesX * context->mmc->itc->ntilesY) *
			(context->tile_output_size * context->tile_output_size) * sizeof(float);
		float *h_inverse_transformed_output = clEnqueueMapBuffer(wrapper.command_queue, context->d_inverse_transformed_output,
			CL_TRUE, CL_MAP_WRITE, 0, len, 0, NULL, NULL, &errcode);
		memcpy(inverse_transformed_output, h_inverse_transformed_output, len);	
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_inverse_transformed_output, h_inverse_transformed_output, 0, NULL, NULL);
	}
}

void free_output_inverse_transform_context(output_inverse_transform_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_AT);
		clReleaseMemObject(context->d_inverse_transformed_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}
#endif