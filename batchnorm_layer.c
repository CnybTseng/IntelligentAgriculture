#include <math.h>
#include "batchnorm_layer.h"
#include "convolutional_layer.h"

void normalize(float *X, float *mean, float *variance, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = X + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] = (at[k] - mean[j]) / (sqrt(variance[j]) + 1e-6);
			}
		}
	}
}

void forward_batchnorm_layer(void *layer, convnet *net)
{
	LAYER_TYPE type = *(LAYER_TYPE *)layer;
	if (type == CONVOLUTIONAL) {
		convolutional_layer *l = (convolutional_layer *)layer;
		int size = l->output_size.w * l->output_size.h;
		if (net->work_mode == INFERENCE) {
			normalize(l->output, l->rolling_mean, l->rolling_variance, l->batch_size, l->nfilters, size);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
			return;
		}
		
		mul_bias(l->output, l->scales, l->batch_size, l->nfilters, size);
		add_bias(l->output, l->biases, l->batch_size, l->nfilters, size);
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void backward_batchnorm_layer(void *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}