#ifndef _RESAMPLE_H_
#define _RESAMPLE_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int stride;
	int batch_size;
	int ninputs;
	int noutputs;
	int upsample;
	float *input;
	float *output;
} resample_layer;

void free_resample_layer(void *_layer);
void print_resample_layer_info(void *_layer, int id);
void set_resample_layer_input(void *_layer, float *input);
float *get_resample_layer_output(void *_layer);
void forward_resample_layer(void *_layer, convnet *net);
void backward_resample_layer(resample_layer *layer, convnet *net);
void upsample(float *in, int width, int height, int nchannels, int stride, float *out);

#ifdef __cplusplus
}
#endif

#endif