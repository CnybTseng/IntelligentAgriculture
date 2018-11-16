#include "maxpool_layer.h"

void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size)
{
	maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
	if (!layer) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = MAXPOOL;
	layer->input_size = input_size;
	layer->filter_size = filter_size;
	layer->stride = stride;
	layer->padding = padding;
	layer->output_size.w = maxpool_output_width(layer);
	layer->output_size.h = maxpool_output_height(layer);
	layer->output_size.c = input_size.c;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->output_size.w * layer->output_size.h * layer->output_size.c;
	layer->input = NULL;
	layer->output = NULL;
	
	if (output_size) {
		*output_size = layer->output_size;
	}
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	
	return (void *)layer;
}					 
						 
void free_maxpool_layer(maxpool_layer *layer)
{
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

int maxpool_output_width(maxpool_layer *layer)
{
	return (layer->input_size.w - layer->filter_size + layer->padding) / layer->stride + 1;
}

int maxpool_output_height(maxpool_layer *layer)
{
	return (layer->input_size.h - layer->filter_size + layer->padding) / layer->stride + 1;
}