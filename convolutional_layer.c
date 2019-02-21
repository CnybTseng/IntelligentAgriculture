#include <omp.h>
#include <math.h>
#include <sys/time.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "activation.h"
#include "zutils.h"
#include "im2col.h"
#include "gemm.h"

#ifdef MERGE_BATCHNORM_TO_CONV
static void merge_batchnorm_params(convolutional_layer *layer);
#endif
#ifdef NNPACK
static void forward_convolutional_layer_nnp(void *_layer, znet *net);
#endif
#ifdef __INTEL_SSE__
static void add_bias_sse(float *output, float *biases, int batch_size, int nchannels, int size);
static void mul_bias_sse(float *output, float *scales, int batch_size, int nchannels, int size);
#elif __ARM_NEON__
static void add_bias_neon(float *output, float *biases, int batch_size, int nchannels, int size);
static void mul_bias_neon(float *output, float *scales, int batch_size, int nchannels, int size);
#endif

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
	layer->transformed_weights = NULL;
	layer->scales = NULL;
	layer->biases = NULL;
	layer->rolling_mean = NULL;
	layer->rolling_variance = NULL;
	layer->input = NULL;
	layer->vecmat = NULL;
	layer->output = NULL;
#ifdef NNPACK
	layer->algorithm = nnp_convolution_algorithm_wt8x8_fp16;
	layer->transformed_kernel_size = 0;
	layer->transformed_kernel = NULL;
#endif
	
	if (output_size) {
		*output_size = layer->output_size;
	}
	
	layer->weights = calloc(layer->nweights, sizeof(float));
	if (!layer->weights) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	if (3 == filter_size) {
		const int tran_size = get_transformed_weight_matrix_size(F6x6_3x3);
		layer->transformed_weights = calloc(tran_size * tran_size * input_size.c * nfilters, sizeof(float));
		if (!layer->transformed_weights) {
			fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
			goto cleanup;
		}
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
		return 0;
	}
	
	return (void *)layer;
}

void free_convolution_layer(void *_layer)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
	if (!layer) return;
	
	if (layer->weights) {
		free(layer->weights);
		layer->weights = NULL;
	}
	
	if (3 == layer->filter_size && layer->transformed_weights) {
		free(layer->transformed_weights);
		layer->transformed_weights = NULL;
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
	
#ifdef NNPACK
	if (layer->transformed_kernel) {
		free(layer->transformed_kernel);
		layer->transformed_kernel = NULL;
	}
#endif	
	
	free(layer);
	layer = NULL;
}

void print_convolutional_layer_info(void *_layer, int id)
{
	convolutional_layer *layer = (convolutional_layer*)_layer;
	static double total_bflop = 0;
	double bflop = layer->filter_size * layer->filter_size * layer->input_size.c * layer->output_size.w *
		layer->output_size.h * layer->output_size.c * 2 / 1000000000.0;
	total_bflop += bflop;
	printf("%2d\tconv\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\t%.9f->%.9f BFLOPs\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->filter_size,
		layer->filter_size,
		layer->stride,
		layer->nfilters,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c,
		bflop,
		total_bflop);
}

void set_convolutional_layer_input(void *_layer, float *input)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
	layer->input = input;
}

float *get_convolutional_layer_output(void *_layer)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
	return layer->output;
}

void forward_convolutional_layer(void *_layer, znet *net)
{
#ifdef NNPACK
	return forward_convolutional_layer_nnp(_layer, net);
#endif	
	convolutional_layer *layer = (convolutional_layer *)_layer;
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
		static double total = 0;
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
		if (1 == layer->filter_size) {
			B = image;
		} else {
			im2col_cpu(image, layer->input_size.w, layer->input_size.h, layer->input_size.c,
				layer->filter_size, layer->stride, layer->padding, B);
		}
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total += duration;
		printf("im2col_cpu: %f ms, total %f ms.\n", duration, total);
		
		gemm(0, 0, m, n, k, 1, A, k, B, n, 1, C, n);
	}

#ifndef MERGE_BATCHNORM_TO_CONV	
	if (layer->batch_norm) {
		forward_batchnorm_layer(layer, net);
	} else {
#endif
		add_bias(layer->output, layer->biases, layer->batch_size, layer->nfilters, n);
#ifndef MERGE_BATCHNORM_TO_CONV
	}
#endif
	
	activate(layer->output, layer->noutputs * layer->batch_size, layer->activation);
}

void backward_convolutional_layer(convolutional_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void load_convolutional_layer_weights(convolutional_layer *layer, FILE *fp)
{
	fread(layer->biases, sizeof(float), layer->nbiases, fp);
	if (layer->batch_norm) {
		fread(layer->scales, sizeof(float), layer->nfilters, fp);
		fread(layer->rolling_mean, sizeof(float), layer->nfilters, fp);
		fread(layer->rolling_variance, sizeof(float), layer->nfilters, fp);
	}
	
	fread(layer->weights, sizeof(float), layer->nweights, fp);
#ifdef MERGE_BATCHNORM_TO_CONV
	if (layer->batch_norm) merge_batchnorm_params(layer);
#endif
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
#ifdef __INTEL_SSE__
	return add_bias_sse(output, biases, batch_size, nchannels, size);
#elif __ARM_NEON__
	return add_bias_neon(output, biases, batch_size, nchannels, size);
#endif
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
#ifdef __INTEL_SSE__
	return mul_bias_sse(output, scales, batch_size, nchannels, size);
#elif __ARM_NEON__
	return mul_bias_neon(output, scales, batch_size, nchannels, size);
#endif
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}

#ifdef MERGE_BATCHNORM_TO_CONV
void merge_batchnorm_params(convolutional_layer *layer)
{
	int num_weis = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->nfilters; ++i) {
		float alpha = layer->scales[i] / sqrt(layer->rolling_variance[i] + 1e-6);
		float *at = layer->weights + i * num_weis;
		for (int j = 0; j < num_weis; ++j) {
			at[j] *= alpha;
		}
		
		layer->biases[i] = layer->biases[i] - layer->rolling_mean[i] * alpha;
	}
}
#endif

#ifdef NNPACK
void forward_convolutional_layer_nnp(void *_layer, znet *net)
{	
	convolutional_layer *layer = (convolutional_layer *)_layer;
	int n = layer->output_size.w * layer->output_size.h;	
	struct nnp_size input_size = {layer->input_size.w, layer->input_size.h};
	struct nnp_padding input_padding = {layer->padding, layer->padding, layer->padding, layer->padding};
	struct nnp_size kernel_size = {layer->filter_size, layer->filter_size};
	struct nnp_size stride = {layer->stride, layer->stride};
	
	float zeros[2048];
	for (int i = 0; i < 2048; ++i) zeros[i] = 0;

	if (3 != layer->filter_size) {
		nnp_convolution_inference(
			nnp_convolution_algorithm_direct,
			nnp_convolution_transform_strategy_tuple_based,
			layer->input_size.c,
			layer->nfilters,
			input_size,
			input_padding,
			kernel_size,
			stride,
			layer->input,
			layer->weights,
			zeros,
			layer->output,
			NULL,
			NULL,
			nnp_activation_identity,
			NULL,
			znet_threadpool(net),
			NULL
		);
	} else {
		if (NULL == layer->transformed_kernel) {
			nnp_convolution_inference(
				layer->algorithm,
				nnp_convolution_transform_strategy_precompute,
				layer->input_size.c,
				layer->nfilters,
				input_size,
				input_padding,
				kernel_size,
				stride,
				NULL,
				NULL,
				NULL,
				NULL,
				NULL,
				&layer->transformed_kernel_size,
				nnp_activation_identity,
				NULL,
				znet_threadpool(net),
				NULL
			);
			
			layer->transformed_kernel = calloc(layer->transformed_kernel_size, 1);
			
			nnp_convolution_inference(
				layer->algorithm,
				nnp_convolution_transform_strategy_precompute,
				layer->input_size.c,
				layer->nfilters,
				input_size,
				input_padding,
				kernel_size,
				stride,
				layer->input,
				layer->weights,
				NULL,
				layer->output,
				layer->transformed_kernel,
				&layer->transformed_kernel_size,
				nnp_activation_identity,
				NULL,
				znet_threadpool(net),
				NULL
			);
		}
		
		static double total = 0;
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
		nnp_convolution_inference(
			layer->algorithm,
			nnp_convolution_transform_strategy_reuse,
			layer->input_size.c,
			layer->nfilters,
			input_size,
			input_padding,
			kernel_size,
			stride,
			layer->input,
			layer->transformed_kernel,
			zeros,
			layer->output,
			NULL,
			NULL,
			nnp_activation_identity,
			NULL,
			znet_threadpool(net),
			NULL
		);
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total += duration;
		printf("nnp_convolution_inference: %f ms, total %f ms.\n", duration, total);
	}

#ifndef MERGE_BATCHNORM_TO_CONV	
	if (layer->batch_norm) {
		forward_batchnorm_layer(layer, net);
	} else {
#endif
		add_bias(layer->output, layer->biases, layer->batch_size, layer->nfilters, n);
#ifndef MERGE_BATCHNORM_TO_CONV
	}
#endif
	
	activate(layer->output, layer->noutputs * layer->batch_size, layer->activation);
}
#endif

#ifdef __INTEL_SSE__
void add_bias_sse(float *output, float *biases, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			__m128 bs = _mm_set1_ps(biases[j]);
			for (int k = 0; k < excess; k += batches) {
				__m128 os = _mm_loadu_ps(at + k);
				os = _mm_add_ps(os, bs);
				_mm_storeu_ps(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

void mul_bias_sse(float *output, float *scales, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			__m128 ss = _mm_set1_ps(scales[j]);
			for (int k = 0; k < excess; k += batches) {
				__m128 os = _mm_loadu_ps(at + k);
				os = _mm_mul_ps(os, ss);
				_mm_storeu_ps(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}

#elif __ARM_NEON__
void add_bias_neon(float *output, float *biases, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			float32x4_t bs = vdupq_n_f32(biases[j]);
			for (int k = 0; k < excess; k += batches) {
				float32x4_t os = vld1q_f32(at + k);
				os = vaddq_f32(os, bs);
				vst1q_f32(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

void mul_bias_neon(float *output, float *scales, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			for (int k = 0; k < excess; k += batches) {
				float32x4_t os = vld1q_f32(at + k);
				os = vmulq_n_f32(os, scales[j]);
				vst1q_f32(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}
#endif