#ifndef _MAXPOOL_LAYER_H_
#define _MAXPOOL_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void forward_maxpool_layer(maxpool_layer *layer, convnet *net);
void backward_maxpool_layer(maxpool_layer *layer, convnet *net);
int maxpool_output_width(maxpool_layer *layer);
int maxpool_output_height(maxpool_layer *layer);

#ifdef __cplusplus
}
#endif

#endif