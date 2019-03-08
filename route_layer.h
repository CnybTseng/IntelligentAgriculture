#ifndef _ROUTE_LAYER_H_
#define _ROUTE_LAYER_H_

#include "znet.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif

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
#ifdef OPENCL
	cl_mem d_output;
#endif
} route_layer;

void free_route_layer(void *_layer);
void print_route_layer_info(void *_layer, int id);
void set_route_layer_input(void *_layer, void *input);
void *get_route_layer_output(void *_layer);
void forward_route_layer(void *_layer, znet *net);
void backward_route_layer(route_layer *layer, znet *net);

#ifdef __cplusplus
}
#endif

#endif