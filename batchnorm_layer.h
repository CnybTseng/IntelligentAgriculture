#ifndef _BATCHNORM_LAYER_H_
#define _BATCHNORM_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void normalize(float *X, float *mean, float *variance, int batch_size, int nchannels, int size);
void forward_batchnorm_layer(void *layer, convnet *net);
void backward_batchnorm_layer(void *layer, convnet *net);

#ifdef __cplusplus
}
#endif

#endif