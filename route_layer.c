#include "route_layer.h"
#include "convolutional_layer.h"
#include "zutils.h"

void *make_route_layer(int batch_size, int *input_layers, int *input_sizes, int nroutes)
{
	route_layer *layer = calloc(1, sizeof(route_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = ROUTE;
	layer->batch_size = batch_size;
	layer->ninputs = 0;
	layer->noutputs = 0;
	layer->nroutes = nroutes;
	layer->input_layers = NULL;
	layer->input_sizes = NULL;
	layer->output = NULL;
	
	layer->input_layers = calloc(nroutes, sizeof(int));
	if (!layer->input_layers) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->input_sizes = calloc(nroutes, sizeof(int));
	if (!layer->input_sizes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	for (int i = 0; i < nroutes; ++i) {
		layer->ninputs += input_sizes[i];
		layer->input_layers[i] = input_layers[i];
		layer->input_sizes[i] = input_sizes[i];
	}

	layer->noutputs = layer->ninputs;
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_route_layer(layer);
	}
	
	return layer;
}

void free_route_layer(route_layer *layer)
{
	if (!layer) return;
	
	if (layer->input_layers) {
		free(layer->input_layers);
		layer->input_layers = NULL;
	}
	
	if (layer->input_sizes) {
		free(layer->input_sizes);
		layer->input_sizes = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

void forward_route_layer(route_layer *layer, convnet *net)
{
	int offset = 0;
	for (int r = 0; r < layer->nroutes; ++r) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[layer->input_layers[r]]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *input_layer = (convolutional_layer *)net->layers[layer->input_layers[r]];
			for (int b = 0; b < layer->batch_size; ++b) {
				float *X = input_layer->output + b * layer->input_sizes[r];
				float *Y = layer->output + b * layer->noutputs + offset;
				mcopy((const char *const)X, (char *const)Y, layer->input_sizes[r] * sizeof(float));
			}
			offset += layer->input_sizes[r];
		} else if (type == RESAMPLE) {
			resample_layer *input_layer = (resample_layer *)net->layers[layer->input_layers[r]];
			for (int b = 0; b < layer->batch_size; ++b) {
				float *X = input_layer->output + b * layer->input_sizes[r];
				float *Y = layer->output + b * layer->noutputs + offset;
				mcopy((const char *const)X, (char *const)Y, layer->input_sizes[r] * sizeof(float));
			}
			offset += layer->input_sizes[r];
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}

void backward_route_layer(route_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}