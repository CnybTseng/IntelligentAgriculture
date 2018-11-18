#include <string.h>
#include "yolo_layer.h"
#include "activation.h"

void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask)
{
	yolo_layer *layer = calloc(1, sizeof(yolo_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = YOLO;
	layer->input_size = input_size;
	layer->output_size = layer->input_size;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->ninputs;
	layer->nscales = nscales;
	layer->total_scales = total_scales;
	layer->classes = classes;
	layer->mask = NULL;
	layer->input = NULL;
	layer->output = NULL;
	
	layer->mask = calloc(nscales, sizeof(int));
	if (!layer->mask) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	for (int i = 0; i < nscales; ++i) {
		layer->mask[i] = mask ? mask[i] : i;
	}
	
	layer->output = calloc(layer->ninputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_yolo_layer(layer);
	}
	
	return (void *)layer;
}

void free_yolo_layer(yolo_layer *layer)
{
	if (!layer) return;
	
	if (layer->mask) {
		free(layer->mask);
		layer->mask = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
}

void forward_yolo_layer(yolo_layer *layer, convnet *net)
{
	int total = layer->ninputs * layer->batch_size;
	memcpy(layer->output, layer->input, total);
	
	if (net->work_mode == INFERENCE) {
		activate(layer->output, total, LOGISTIC);
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void backward_yolo_layer(yolo_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}