#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "winograd_convolution.h"
#include "gemm.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

#define MAX_FILTER_CHANNELS	1024

#ifdef OPENCL
extern cl_wrapper wrapper;

struct weight_transform_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_weight;
	cl_mem d_transformed_weight;
	int transf_weight_size;
};

struct input_transform_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_transformed_input;
	int input_width;
	int input_height;
	int input_channels;
	int stride;
	int padding;
};

struct winograd_convolution_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_output;
};
#endif

int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv)
{
	int filter_size;
	int conv_tile_output_size;
	if (conv == F6x6_3x3) {
		filter_size = 3;
		conv_tile_output_size = 6;
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	return filter_size + conv_tile_output_size - 1;
}

#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv)
{
	weight_transform_context *context = calloc(1, sizeof(weight_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	cl_int errcode;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "weight_transform_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	context->d_weight = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		3 * 3 * MAX_FILTER_CHANNELS * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	context->transf_weight_size = get_transformed_weight_matrix_size(conv);
	context->d_transformed_weight = clCreateBuffer(wrapper.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		context->transf_weight_size * context->transf_weight_size * MAX_FILTER_CHANNELS *
		sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		clean:free_weight_transform_context(context);
		return 0;
	}
		
	return context;
}

void transform_weight(weight_transform_context *context, float *weights, int filter_size,
                      int filter_channels, int nfilters, float *transformed_weights)
{
	cl_int errcode;
	const int standard_batch_size = MAX_FILTER_CHANNELS;
	int num_unprocessed_channels = filter_channels * nfilters;
	const int num_batches = (num_unprocessed_channels + 1023) >> 10;
	static int counter = 0;
	printf("...num_batches=%d...", num_batches);
#ifdef CL_PROFILING_ENABLE	
	float duration = 0;
#endif
	for (int b = 0; b < num_batches; ++b) {
		const int transf_weight_size = context->transf_weight_size;
		int batch_size = num_unprocessed_channels < standard_batch_size ? num_unprocessed_channels : standard_batch_size;
		
		float *h_weight = clEnqueueMapBuffer(wrapper.command_queue, context->d_weight, CL_TRUE, CL_MAP_WRITE,
			0, filter_size * filter_size * batch_size * sizeof(float), 0, NULL, NULL, &errcode);
		memcpy(h_weight, weights + b * standard_batch_size * filter_size * filter_size, filter_size * filter_size *
			batch_size * sizeof(float));
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_weight, h_weight, 0, NULL, NULL);

		errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_weight); 
		errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_transformed_weight); 
		errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_int), &filter_channels);
		errcode |= clSetKernelArg(context->kernel, 3, sizeof(cl_int), &nfilters);
		
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
		
		float *h_transformed_weight = clEnqueueMapBuffer(wrapper.command_queue, context->d_transformed_weight,
			CL_TRUE, CL_MAP_READ, 0, transf_weight_size * transf_weight_size * batch_size *
			sizeof(float), 0, NULL, NULL, &errcode);
		memcpy(transformed_weights + b * standard_batch_size * transf_weight_size * transf_weight_size, h_transformed_weight,
			transf_weight_size * transf_weight_size * batch_size * sizeof(float));
		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_weight, h_transformed_weight, 0, NULL, NULL);
		num_unprocessed_channels -= batch_size;
	}
#ifdef CL_PROFILING_ENABLE
	printf("...%fms...", duration);
#endif
	if (counter++ == 3) {
		save_volume(weights, 3, 3, filter_channels * nfilters, "weights.txt");
		save_volume(transformed_weights, 8, 8, filter_channels * nfilters, "transformed_weights.txt");
	}
}

void free_weight_transform_context(weight_transform_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_weight);
		clReleaseMemObject(context->d_transformed_weight);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

input_transform_context *create_input_transform_context(int input_width, int input_height,
	int input_channels, int stride, int padding)
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
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "input_transform_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	image_desc.image_width = input_width >> 2;
	image_desc.image_height = input_height;
	image_desc.image_depth = input_channels;
	context->d_input = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	const int ntilesX = (input_width / 6) + ((input_width % 6) > 1);
	const int ntilesY = (input_height / 6) + ((input_height % 6) > 1);
	
	flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
	image_format.image_channel_order = CL_DEPTH;
	image_format.image_channel_data_type = CL_FLOAT;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
	image_desc.image_width = ntilesX * ntilesY;
	image_desc.image_height = input_channels;
	image_desc.image_depth = 64;
	context->d_transformed_input = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		clean:free_input_transform_context(context);
	}
	printf("transformed input tensor size %dx%dx%d\n", image_desc.image_height, image_desc.image_width, image_desc.image_depth);
	return context;
}

void transform_input(input_transform_context *context, float *input)
{
	
}

void free_input_transform_context(input_transform_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_input);
		clReleaseMemObject(context->d_transformed_input);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

winograd_convolution_context *create_winograd_convolution_context(WINOGRAD_CONV_TYPE conv,
	int input_width, int input_height, int input_channels, int stride, int padding, int output_channels)
{
	winograd_convolution_context *context = calloc(1, sizeof(winograd_convolution_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	cl_int errcode;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "winograd_convolution_f6x6_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	image_desc.image_width = input_width >> 2;
	image_desc.image_height = input_height;
	image_desc.image_depth = input_channels;
	context->d_input = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	image_desc.image_width = ((input_width + 2 * padding - 3) / stride + 1) >> 2;
	image_desc.image_height = (input_height + 2 * padding - 3) / stride + 1;
	image_desc.image_depth = output_channels;
	context->d_output = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		clean:free_winograd_convolution_context(context);
	}
	
	return context;
}

void winograd_convolution(weight_transform_context *wt_context, winograd_convolution_context *wc_context,
	float *input, int input_width, int input_height, int input_channels, int stride, int padding,
	int output_channels, float *output)
{
	
}						  
						  
void free_winograd_convolution_context(winograd_convolution_context *context)
{
	if (context) {
		clReleaseMemObject(context->d_input);
		clReleaseMemObject(context->d_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}
#endif