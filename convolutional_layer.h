#ifndef _CONVOLUTIONAL_LAYER_H_
#define _CONVOLUTIONAL_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void print_convolutional_layer_info(void *_layer, int id);
void set_convolutional_layer_input(void *_layer, float *input);
float *get_convolutional_layer_output(void *_layer);
void forward_convolutional_layer(void *_layer, convnet *net);
void backward_convolutional_layer(convolutional_layer *layer, convnet *net);
void load_convolutional_layer_weights(convolutional_layer *layer, FILE *fp);
int convolutional_output_width(convolutional_layer *layer);
int convolutional_output_height(convolutional_layer *layer);
void add_bias(float *output, float *biases, int batch_size, int nchannels, int size);
void mul_bias(float *output, float *scales, int batch_size, int nchannels, int size);

#ifdef __cplusplus
}
#endif

#endif