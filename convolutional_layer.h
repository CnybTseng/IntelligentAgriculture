#ifndef _CONVOLUTIONAL_LAYER_H_
#define _CONVOLUTIONAL_LAYER_H_

#include "znet.h"
#include "winograd_convolution.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef OPENCL
struct direct_convolution_context;
typedef struct direct_convolution_context direct_convolution_context;
#endif

typedef struct {
	LAYER_TYPE type;
	ACTIVATION activation;
	dim3 input_size;
	dim3 output_size;
	int filter_size;
	int nfilters;
	int stride;
	int padding;
	int batch_size;
	int batch_norm;
	int nweights;
	int nbiases;
	int ninputs;
	int vmsize;
	int noutputs;
	float *weights;
	float *scales;
	float *biases;
	float *rolling_mean;
	float *rolling_variance;
	float *input;
	float *vecmat;
	float *output;
#ifdef NNPACK
	enum nnp_convolution_algorithm algorithm;
	size_t transformed_kernel_size;
	float *transformed_kernel;
#endif
#if defined(OPENCL)
#ifdef WINOGRAD_CONVOLUTION
	weight_transform_context *wtc;
	input_transform_context *itc;
	matrix_multiplication_context *mmc;
	output_inverse_transform_context *oitc;
#endif
	direct_convolution_context *dcc;
#endif
} convolutional_layer;

void free_convolution_layer(void *_layer);
void print_convolutional_layer_info(void *_layer, int id);
void set_convolutional_layer_input(void *_layer, void *input);
void *get_convolutional_layer_output(void *_layer);
void forward_convolutional_layer(void *_layer, znet *net);
void backward_convolutional_layer(convolutional_layer *layer, znet *net);
void load_convolutional_layer_weights(convolutional_layer *layer, FILE *fp);
int convolutional_output_width(convolutional_layer *layer);
int convolutional_output_height(convolutional_layer *layer);
void add_bias(float *output, float *biases, int batch_size, int nchannels, int size);
void mul_bias(float *output, float *scales, int batch_size, int nchannels, int size);
#ifdef OPENCL
void get_direct_convolution_output_image_size(convolutional_layer *layer, int *width, int *height);
#endif

#ifdef __cplusplus
}
#endif

#endif