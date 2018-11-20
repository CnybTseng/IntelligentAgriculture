#ifndef _MAXPOOL_LAYER_H_
#define _MAXPOOL_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void print_maxpool_layer_info(void *_layer, int id);
void set_maxpool_layer_input(void *_layer, float *input);
float *get_maxpool_layer_output(void *_layer);
void forward_maxpool_layer(void *_layer, convnet *net);
void backward_maxpool_layer(maxpool_layer *layer, convnet *net);
int maxpool_output_width(maxpool_layer *layer);
int maxpool_output_height(maxpool_layer *layer);

#ifdef __cplusplus
}
#endif

#endif