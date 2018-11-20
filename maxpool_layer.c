#include <float.h>
#include "maxpool_layer.h"

void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size)
{
	maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
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
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free_maxpool_layer(layer);
	}
	
	return (void *)layer;
}					 
						 
void free_maxpool_layer(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

void print_maxpool_layer_info(void *_layer, int id)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	printf("%2d\tmaxpool\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->filter_size,
		layer->filter_size,
		layer->stride,
		layer->input_size.c,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c);
}

void set_maxpool_layer_input(void *_layer, float *input)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	layer->input = input;
}

float *get_maxpool_layer_output(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	return layer->output;
}

void forward_maxpool_layer(void *_layer, convnet *net)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	int offsetx = -layer->padding / 2;
	int offsety = -layer->padding / 2;
	int inwh = layer->input_size.w * layer->input_size.h;
	int outwh = layer->output_size.w * layer->output_size.h;
	
	for (int b = 0; b < layer->batch_size; ++b) {
		for (int c = 0; c < layer->output_size.c; ++c) {
			int dslice = (b * layer->output_size.c + c) * outwh;
			int dslice0 = (b * layer->input_size.c + c) * inwh;
			for (int y = 0; y < layer->output_size.h; ++y) {
				for (int x = 0; x < layer->output_size.w; ++x) {
					int maxidx = -1;
					float maxval = -FLT_MAX;
					for (int dy = 0; dy < layer->filter_size; ++dy) {
						for (int dx = 0; dx < layer->filter_size; ++dx) {
							int x0 = x * layer->stride + dx + offsetx;
							int y0 = y * layer->stride + dy + offsety;
							int idx0 = dslice0 + y0 * layer->input_size.w + x0;
							int valid = x0 > -1 && x0 < layer->input_size.w &&
								y0 > -1 && y0 < layer->input_size.h;
							float val = valid ? layer->input[idx0] : -FLT_MAX;
							int bigger = val > maxval;
							maxidx = bigger ? idx0 : maxidx;
							maxval = bigger ? val : maxval;
						}
					}
					
					int idx = dslice + y * layer->output_size.w + x;
					layer->output[idx] = maxval;
				}
			}
		}
	}
}

void backward_maxpool_layer(maxpool_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int maxpool_output_width(maxpool_layer *layer)
{
	return (layer->input_size.w - layer->filter_size + layer->padding) / layer->stride + 1;
}

int maxpool_output_height(maxpool_layer *layer)
{
	return (layer->input_size.h - layer->filter_size + layer->padding) / layer->stride + 1;
}