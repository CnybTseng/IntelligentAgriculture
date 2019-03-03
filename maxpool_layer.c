#include <omp.h>
#include <float.h>
#include <string.h>
#include "maxpool_layer.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

#ifdef NNPACK
struct maxpool_thread_param {
	void *layer;
	znet *net;
};

static void maxpool_thread(struct maxpool_thread_param *param, size_t batch_size, size_t nchannels);
#endif

#ifdef OPENCL
extern cl_wrapper wrapper;
void cl_forward_maxpool_layer(void *_layer, znet *net);
#endif

void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size)
{
	maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = MAXPOOL;
	layer->input_size = input_size;
	layer->filter_size = filter_size;
	layer->stride = stride;
	layer->padding = padding;
	layer->output_size.w = maxpool_output_width(layer);
	layer->output_size.h = maxpool_output_height(layer);
	layer->output_size.c = input_size.c;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->output_size.w * layer->output_size.h * layer->output_size.c;
	layer->input = NULL;
	layer->output = NULL;
	
	if (output_size) {
		*output_size = layer->output_size;
	}
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free_maxpool_layer(layer);
	}
	
	return (void *)layer;
}					 
						 
void free_maxpool_layer(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

void print_maxpool_layer_info(void *_layer, int id)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	printf("%2d\tmaxpool\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->filter_size,
		layer->filter_size,
		layer->stride,
		layer->input_size.c,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c);
}

void set_maxpool_layer_input(void *_layer, float *input)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	layer->input = input;
}

float *get_maxpool_layer_output(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	return layer->output;
}

void forward_maxpool_layer(void *_layer, znet *net)
{	
	maxpool_layer *layer = (maxpool_layer *)_layer;
#ifdef NNPACK
	struct maxpool_thread_param param = {_layer, net};
	return pthreadpool_compute_2d(znet_threadpool(net), (pthreadpool_function_2d_t)maxpool_thread,
		&param, layer->batch_size, layer->output_size.c);
#endif
#ifdef OPENCL
	return cl_forward_maxpool_layer(_layer, net);
#endif
	int offsetx = -layer->padding / 2;
	int offsety = -layer->padding / 2;
	int inwh = layer->input_size.w * layer->input_size.h;
	int outwh = layer->output_size.w * layer->output_size.h;
	
	for (int b = 0; b < layer->batch_size; ++b) {
		#pragma omp parallel for
		for (int c = 0; c < layer->output_size.c; ++c) {
			int dslice = (b * layer->output_size.c + c) * outwh;
			int dslice0 = (b * layer->input_size.c + c) * inwh;
			for (int y = 0; y < layer->output_size.h; ++y) {
				for (int x = 0; x < layer->output_size.w; ++x) {
					int maxidx = -1;
					float maxval = -FLT_MAX;
					for (int dy = 0; dy < layer->filter_size; ++dy) {
						for (int dx = 0; dx < layer->filter_size; ++dx) {
							int x0 = x * layer->stride + dx + offsetx;
							int y0 = y * layer->stride + dy + offsety;
							int idx0 = dslice0 + y0 * layer->input_size.w + x0;
							int valid = x0 > -1 && x0 < layer->input_size.w &&
								y0 > -1 && y0 < layer->input_size.h;
							float val = valid ? layer->input[idx0] : -FLT_MAX;
							int bigger = val > maxval;
							maxidx = bigger ? idx0 : maxidx;
							maxval = bigger ? val : maxval;
						}
					}
					
					int idx = dslice + y * layer->output_size.w + x;
					layer->output[idx] = maxval;
				}
			}
		}
	}
}

void backward_maxpool_layer(maxpool_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int maxpool_output_width(maxpool_layer *layer)
{
	return (layer->input_size.w - layer->filter_size + layer->padding) / layer->stride + 1;
}

int maxpool_output_height(maxpool_layer *layer)
{
	return (layer->input_size.h - layer->filter_size + layer->padding) / layer->stride + 1;
}

#ifdef NNPACK
void maxpool_thread(struct maxpool_thread_param *param, size_t batch_size, size_t nchannels)
{
	maxpool_layer *layer = (maxpool_layer *)param->layer;
	int offsetx = -layer->padding / 2;
	int offsety = -layer->padding / 2;
	int inwh = layer->input_size.w * layer->input_size.h;
	int outwh = layer->output_size.w * layer->output_size.h;
	
	int dslice = (batch_size * layer->output_size.c + nchannels) * outwh;
	int dslice0 = (batch_size * layer->input_size.c + nchannels) * inwh;
	for (int y = 0; y < layer->output_size.h; ++y) {
		for (int x = 0; x < layer->output_size.w; ++x) {
			int maxidx = -1;
			float maxval = -FLT_MAX;
			for (int dy = 0; dy < layer->filter_size; ++dy) {
				for (int dx = 0; dx < layer->filter_size; ++dx) {
					int x0 = x * layer->stride + dx + offsetx;
					int y0 = y * layer->stride + dy + offsety;
					int idx0 = dslice0 + y0 * layer->input_size.w + x0;
					int valid = x0 > -1 && x0 < layer->input_size.w &&
						y0 > -1 && y0 < layer->input_size.h;
					float val = valid ? layer->input[idx0] : -FLT_MAX;
					int bigger = val > maxval;
					maxidx = bigger ? idx0 : maxidx;
					maxval = bigger ? val : maxval;
				}
			}
			
			int idx = dslice + y * layer->output_size.w + x;
			layer->output[idx] = maxval;
		}
	}
}
#endif

#ifdef OPENCL
void cl_forward_maxpool_layer(void *_layer, znet *net)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	
	cl_program program = 0;
	cl_kernel kernel = 0;
	cl_mem d_input = 0;
	cl_mem d_output = 0;
	const int channel_blocks = (layer->input_size.c + 3) >> 2;
	const int input_image_width = layer->input_size.w * channel_blocks;
	const int input_image_height = layer->input_size.h;
	const int output_image_width = layer->output_size.w * channel_blocks;
	const int output_image_height = layer->output_size.h;
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	char options[] = "";
	program = cl_make_wrapper_program(wrapper, "maxpool.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	kernel = cl_make_wrapper_kernel(wrapper, program, "maxpool_2x2", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = input_image_width;
	image_desc.image_height = input_image_height;
	
	d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = output_image_width;
	image_desc.image_height = output_image_height;
	
	d_output = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	cl_event event;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {input_image_width, input_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	float *h_input = clEnqueueMapImage(wrapper.command_queue, d_input, CL_TRUE, CL_MAP_WRITE,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	nchw_to_nhwc_quad(layer->input, h_input, layer->input_size.w, layer->input_size.h, layer->input_size.c, 1);
	// save_volume(h_input, input_image_width << 2, input_image_height, 1, "formated_input.txt");
	clEnqueueUnmapMemObject(wrapper.command_queue, d_input, h_input, 0, NULL, &event);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
	errcode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &layer->input_size.w);
	errcode |= clSetKernelArg(kernel, 3, sizeof(int), &layer->input_size.h);
	errcode |= clSetKernelArg(kernel, 4, sizeof(int), &layer->output_size.w);
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &layer->output_size.h);
	errcode |= clSetKernelArg(kernel, 6, sizeof(int), &layer->padding);
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &layer->stride);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	printf("global work size %dx%d\n", output_image_width, output_image_height);

	cl_uint work_dim = 2;
	size_t global_work_size[] = {output_image_width, output_image_height};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	printf("GPU, maxpool: %fms.\n", duration);
#endif
	clReleaseEvent(event);

	region[0] = output_image_width;
	region[1] = output_image_height;
	float *h_output = clEnqueueMapImage(wrapper.command_queue, d_output, CL_TRUE, CL_MAP_READ,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clEnqueueMapImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	image_row_pitch = image_row_pitch >> 2;
	for (int y = 0; y < output_image_height; ++y) {
		memcpy(layer->output + y * layer->output_size.w * layer->output_size.c, h_output + y * image_row_pitch,
			layer->output_size.w * layer->output_size.c * sizeof(float));
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_output, h_output, 0, NULL, &event);

	cleanup:
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	cl_destroy_wrapper(wrapper);
}
#endif