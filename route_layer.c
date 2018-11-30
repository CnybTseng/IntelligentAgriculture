#include <stdarg.h>
#include "route_layer.h"
#include "convolutional_layer.h"
#include "resample_layer.h"
#include "zutils.h"

static int parse_input_layer(void *layer, dim3 *output_size);

void *make_route_layer(int batch_size, int nroutes, void *layers[], int *layer_id, dim3 *output_size)
{
	route_layer *layer = calloc(1, sizeof(route_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = ROUTE;
	layer->output_size.w = 0;
	layer->output_size.h = 0;
	layer->output_size.c = 0;
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
		layer->input_layers[i] = layer_id[i];
		layer->input_sizes[i] = parse_input_layer(layers[i], &layer->output_size);
		layer->ninputs += layer->input_sizes[i];
	}
	
	if (output_size) {
		*output_size = layer->output_size;
	}

	layer->noutputs = layer->ninputs;
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_route_layer(layer);
	}
	
	return layer;
}

void free_route_layer(void *_layer)
{
	route_layer *layer = (route_layer *)_layer;
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

void print_route_layer_info(void *_layer, int id)
{
	route_layer *layer = (route_layer *)_layer;
	printf("%02d\troute ", id);
	for (int i = 0; i < layer->nroutes; ++i) {
		printf("%d", layer->input_layers[i] + 1);
		if (i < layer->nroutes - 1) printf(",");
	}
	printf("\n");
}

void set_route_layer_input(void *_layer, float *input)
{
	;
}

float *get_route_layer_output(void *_layer)
{
	route_layer *layer = (route_layer *)_layer;
	return layer->output;
}

void forward_route_layer(void *_layer, znet *net)
{
	route_layer *layer = (route_layer *)_layer;
	int offset = 0;
	void **layers = znet_layers(net);
	for (int r = 0; r < layer->nroutes; ++r) {
		LAYER_TYPE type = *(LAYER_TYPE *)(layers[layer->input_layers[r]]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *input_layer = (convolutional_layer *)layers[layer->input_layers[r]];
			for (int b = 0; b < layer->batch_size; ++b) {
				float *X = input_layer->output + b * layer->input_sizes[r];
				float *Y = layer->output + b * layer->noutputs + offset;
				mcopy((const char *const)X, (char *const)Y, layer->input_sizes[r] * sizeof(float));
			}
			offset += layer->input_sizes[r];
		} else if (type == RESAMPLE) {
			resample_layer *input_layer = (resample_layer *)layers[layer->input_layers[r]];
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

void backward_route_layer(route_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int parse_input_layer(void *layer, dim3 *output_size)
{
	LAYER_TYPE type = *(LAYER_TYPE *)layer;
	if (type == CONVOLUTIONAL) {
		convolutional_layer *l = (convolutional_layer *)layer;
		output_size->w  = l->output_size.w;
		output_size->h  = l->output_size.h;
		output_size->c += l->output_size.c;
		return l->noutputs;
	} else if (type == RESAMPLE) {
		resample_layer *l = (resample_layer *)layer;
		output_size->w  = l->output_size.w;
		output_size->h  = l->output_size.h;
		output_size->c += l->output_size.c;
		return l->noutputs;
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
}