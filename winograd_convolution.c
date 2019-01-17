#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "winograd_convolution.h"
#include "gemm.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

#ifdef OPENCL
extern cl_wrapper wrapper;

struct weight_transform_context {
	cl_program program;
	cl_kernel kernel;
	cl_mem d_weight;
	cl_mem d_transformed_weight;
	int transf_weight_size;
	int standard_batch_size;
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
	struct weight_transform_context *context = calloc(1, sizeof(struct weight_transform_context));
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
	
	const int standard_batch_size = 1024;
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;	
	cl_image_format image_format = {CL_RGBA, CL_FLOAT};
	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE3D,
		.image_width = 1,
		.image_height = 4,
		.image_depth = standard_batch_size,
		.image_row_pitch = 0,
		.image_slice_pitch = 0};	

	context->d_weight = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto clean;
	}
	
	flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
	image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	image_desc.image_width = 2;
	image_desc.image_height = 8;
	image_desc.image_depth = standard_batch_size;
	image_desc.image_row_pitch = 0,
	image_desc.image_slice_pitch = 0;

	context->d_transformed_weight = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		clean:free_weight_transform_context(context);
		return 0;
	}
	
	context->transf_weight_size = get_transformed_weight_matrix_size(conv);
	context->standard_batch_size = standard_batch_size;
	
	return context;
}

void transform_weight(weight_transform_context *context, float *weights, int filter_size,
                      int filter_channels, int nfilters, float *transformed_weights)
{
	static int counter = 0;
	cl_event event;
	cl_int errcode;
	const int transf_weight_size = context->transf_weight_size;
	const int standard_batch_size = context->standard_batch_size;
	int num_unprocessed_channels = filter_channels * nfilters;
	const int num_batches = (num_unprocessed_channels + 1023) >> 10;
	
	for (int b = 0; b < num_batches; ++b) {
		size_t origin[] = {0, 0, 0};
		size_t region[] = {0, 0, 0};
		clGetImageInfo(context->d_weight, CL_IMAGE_WIDTH,  sizeof(size_t), region,     NULL);
		clGetImageInfo(context->d_weight, CL_IMAGE_HEIGHT, sizeof(size_t), region + 1, NULL);
		clGetImageInfo(context->d_weight, CL_IMAGE_DEPTH,  sizeof(size_t), region + 2, NULL);
		size_t image_row_pitch, image_slice_pitch;
		float *h_weight = clEnqueueMapImage(wrapper.command_queue, context->d_weight, CL_TRUE, CL_MAP_WRITE, origin,
			region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);

		int batch_size = num_unprocessed_channels < standard_batch_size ? num_unprocessed_channels : standard_batch_size;
		image_row_pitch = image_row_pitch >> 2;
		image_slice_pitch = image_slice_pitch >> 2;
		
		for (int i = 0; i < batch_size; ++i) {
			float *to = h_weight + i * image_slice_pitch;
			float *fr = weights + (b * standard_batch_size + i) * filter_size * filter_size;
			for (int j = 0; j < filter_size; ++j) {
				to[j * image_row_pitch]     = fr[j * filter_size];
				to[j * image_row_pitch + 1] = fr[j * filter_size + 1];
				to[j * image_row_pitch + 2] = fr[j * filter_size + 2];
				to[j * image_row_pitch + 3] = 0;
			}
			
			to[filter_size * image_row_pitch]     = 0;
			to[filter_size * image_row_pitch + 1] = 0;
			to[filter_size * image_row_pitch + 2] = 0;
			to[filter_size * image_row_pitch + 3] = 0;
		}

		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_weight, h_weight, 0, NULL, &event);

		errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_weight); 
		errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_transformed_weight); 	
		if (CL_SUCCESS != errcode) {
			fprintf(stderr, "clSetKernelArg fail!\n");
			return;
		}

		cl_uint work_dim = 3;
		size_t global_work_size[] = {1, 1, batch_size};
		size_t local_work_size[]  = {1, 1, standard_batch_size};
		clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
			local_work_size, 0, NULL, &event);

		clGetImageInfo(context->d_transformed_weight, CL_IMAGE_WIDTH, sizeof(size_t), region, NULL);
		clGetImageInfo(context->d_transformed_weight, CL_IMAGE_HEIGHT, sizeof(size_t), region + 1, NULL);
		clGetImageInfo(context->d_transformed_weight, CL_IMAGE_DEPTH, sizeof(size_t), region + 2, NULL);
		float *h_transformed_weight = clEnqueueMapImage(wrapper.command_queue, context->d_transformed_weight, CL_TRUE,
			CL_MAP_WRITE, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);

		image_row_pitch = image_row_pitch >> 2;
		image_slice_pitch = image_slice_pitch >> 2;
		
		for (int i = 0; i < batch_size; ++i) {
			float *to = transformed_weights + (b * standard_batch_size + i) * transf_weight_size * transf_weight_size;
			float *fr = h_transformed_weight + i * image_slice_pitch;
			for (int j = 0; j < transf_weight_size; ++j) {
				memcpy(to + j * transf_weight_size, fr + j * image_row_pitch, transf_weight_size * sizeof(float));
			}
		}

		clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_weight, h_transformed_weight, 0, NULL, &event);
		num_unprocessed_channels -= batch_size;
	}

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
#endif