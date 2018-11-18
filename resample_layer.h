#ifndef _RESAMPLE_H_
#define _RESAMPLE_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void forward_upsample_layer(resample_layer *layer, convnet *net);
void backward_upsample_layer(resample_layer *layer, convnet *net);
void upsample(float *in, int width, int height, int nchannels, int stride, float *out);

#ifdef __cplusplus
}
#endif

#endif