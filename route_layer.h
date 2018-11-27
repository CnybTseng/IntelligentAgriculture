#ifndef _ROUTE_LAYER_H_
#define _ROUTE_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	LAYER_TYPE type;
	dim3 output_size;
	int batch_size;
	int ninputs;
	int noutputs;
	int nroutes;
	int *input_layers;
	int *input_sizes;
	float *output;
} route_layer;

void free_route_layer(void *_layer);
void print_route_layer_info(void *_layer, int id);
void set_route_layer_input(void *_layer, float *input);
float *get_route_layer_output(void *_layer);
void forward_route_layer(void *_layer, convnet *net);
void backward_route_layer(route_layer *layer, convnet *net);

#ifdef __cplusplus
}
#endif

#endif