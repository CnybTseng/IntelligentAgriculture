#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "activation.h"
#include "zutils.h"
#include "im2col.h"
#include "gemm.h"

void *make_convolutional_layer(ACTIVATION activation, dim3 input_size, int filter_size, int nfilters,
                               int stride, int padding, int batch_size, int batch_norm, dim3 *output_size)
{
	convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = CONVOLUTIONAL;
	layer->activation = activation;
	layer->input_size = input_size;
	layer->filter_size = filter_size;
	layer->nfilters = nfilters;
	layer->stride = stride;
	layer->padding = padding;
	layer->output_size.w = convolutional_output_width(layer);
	layer->output_size.h = convolutional_output_height(layer);
	layer->output_size.c = nfilters;
	layer->batch_size = batch_size;
	layer->batch_norm = batch_norm;
	layer->nweights = filter_size * filter_size * input_size.c * nfilters;
	layer->nbiases = nfilters;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->vmsize = input_size.c * filter_size * filter_size * layer->output_size.w * layer->output_size.h;
	layer->noutputs = layer->output_size.w * layer->output_size.h * nfilters;
	layer->weights = NULL;
	layer->scales = NULL;
	layer->biases = NULL;
	layer->rolling_mean = NULL;
	layer->rolling_variance = NULL;
	layer->input = NULL;
	layer->vecmat = NULL;
	layer->output = NULL;
	
	if (output_size) {
		*output_size = layer->output_size;
	}
	
	layer->weights = calloc(layer->nweights, sizeof(float));
	if (!layer->weights) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->scales = calloc(nfilters, sizeof(float));
	if (!layer->scales) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->biases = calloc(layer->nbiases, sizeof(float));
	if (!layer->biases) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->rolling_mean = calloc(nfilters, sizeof(float));
	if (!layer->rolling_mean) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->rolling_variance = calloc(nfilters, sizeof(float));
	if (!layer->rolling_variance) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->vecmat = calloc(layer->vmsize, sizeof(float));
	if (!layer->vecmat) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_convolution_layer(layer);
	}
	
	return (void *)layer;
}

void free_convolution_layer(convolutional_layer *layer)
{
	if (!layer) return;
	
	if (layer->weights) {
		free(layer->weights);
		layer->weights = NULL;
	}
	
	if (layer->scales) {
		free(layer->scales);
		layer->scales = NULL;
	}
	
	if (layer->biases) {
		free(layer->biases);
		layer->biases = NULL;
	}
	
	if (layer->rolling_mean) {
		free(layer->rolling_mean);
		layer->rolling_mean = NULL;
	}
	
	if (layer->rolling_variance) {
		free(layer->rolling_variance);
		layer->rolling_variance = NULL;
	}
	
	if (layer->vecmat) {
		free(layer->vecmat);
		layer->vecmat = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

void forward_convolutional_layer(convolutional_layer *layer, convnet *net)
{
	float alpha = 0;
	size_t size = layer->noutputs * layer->batch_size * sizeof(float);
	mset((char *const)layer->output, size, (const char *const)&alpha, sizeof(float));
	
	int m = layer->nfilters;
	int n = layer->output_size.w * layer->output_size.h;
	int k = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->batch_size; ++i) {
		float *image = layer->input + i * layer->ninputs;
		float *A = layer->weights;
		float *B = layer->vecmat;
		float *C = layer->output + i * layer->noutputs;
		if (1 == layer->filter_size) {
			B = image;
		} else {
			im2col_cpu(image, layer->input_size.w, layer->input_size.h, layer->input_size.c,
				layer->filter_size, layer->stride, layer->padding, B);
		}
		
		gemm(0, 0, m, n, k, 1, A, k, B, n, 1, C, n);
	}
	
	if (layer->batch_norm) {
		forward_batchnorm_layer(layer, net);
	} else {
		add_bias(layer->output, layer->biases, layer->batch_size, layer->nfilters, n);
	}
	
	activate(layer->output, layer->noutputs * layer->batch_size, layer->activation);
}

void backward_convolutional_layer(convolutional_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int convolutional_output_width(convolutional_layer *layer)
{
	return (layer->input_size.w + 2 * layer->padding - layer->filter_size) / layer->stride + 1;
}

int convolutional_output_height(convolutional_layer *layer)
{
	return (layer->input_size.h + 2 * layer->padding - layer->filter_size) / layer->stride + 1;
}

/** @brief 添加加性偏置到输入的卷积输出,或已添加乘性偏置的输入的卷积输出上.
 ** @param output 输入的卷积输出,或已添加乘性偏置的输入的卷积输出.
 ** @param biases 神经元加性偏置.
 ** @param batch_size 批量大小.
 ** @param nchannels 卷积输出的通道数.
 ** @param size 卷积输出的大小.
 **/
void add_bias(float *output, float *biases, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

/** @brief 添加乘性偏置到输入的卷积输出上.
 ** @param output 输入的卷积输出.
 ** @param scales 神经元乘性偏置.
 ** @param batch_size 批量大小.
 ** @param nchannels 卷积输出的通道数.
 ** @param size 卷积输出的大小.
 **/
void mul_bias(float *output, float *scales, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}