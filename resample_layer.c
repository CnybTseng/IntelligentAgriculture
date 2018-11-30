#include "resample_layer.h"
#include "zutils.h"

void *make_resample_layer(dim3 input_size, int batch_size, int stride, dim3 *output_size)
{
	resample_layer *layer = calloc(1, sizeof(resample_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = RESAMPLE;
	layer->input_size = input_size;
	if (stride > 0) {
		layer->output_size.w = input_size.w * stride;
		layer->output_size.h = input_size.h * stride;
		layer->stride = stride;
		layer->upsample = 1;
	} else {
		layer->output_size.w = input_size.w / stride;
		layer->output_size.h = input_size.h / stride;
		layer->stride = -stride;
		layer->upsample = 0;
	}
	
	layer->output_size.c = input_size.c;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->output_size.w * layer->output_size.h * layer->output_size.c;
	layer->input = NULL;
	layer->output = NULL;
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free_resample_layer(layer);
	}
	
	return layer;
}

void free_resample_layer(void *_layer)
{
	resample_layer *layer = (resample_layer *)_layer;
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
	free(layer);
	layer = NULL;
}

void print_resample_layer_info(void *_layer, int id)
{
	resample_layer *layer = (resample_layer *)_layer;
	printf("%2d\tresample\t%4d x%4d x%4d\t\t%d\t\t\t\t%4d x%4d x%4d\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->stride,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c);
}

void set_resample_layer_input(void *_layer, float *input)
{
	resample_layer *layer = (resample_layer *)_layer;
	layer->input = input;
}

float *get_resample_layer_output(void *_layer)
{
	resample_layer *layer = (resample_layer *)_layer;
	return layer->output;
}

void forward_resample_layer(void *_layer, znet *net)
{
	resample_layer *layer = (resample_layer *)_layer;
	float alpha = 0;
	size_t size = layer->noutputs * layer->batch_size * sizeof(float);
	mset((char *const)layer->output, size, (const char *const)&alpha, sizeof(float));
	
	for (int b = 0; b < layer->batch_size; ++b) {
		float *in = layer->input + b * layer->ninputs;
		float *out = layer->output + b * layer->noutputs;
		if (layer->upsample) {
			upsample(in, layer->input_size.w, layer->input_size.h,
				layer->input_size.c, layer->stride, out);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}

void backward_resample_layer(resample_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void upsample(float *in, int width, int height, int nchannels, int stride, float *out)
{
	int us_width = width * stride;
	int us_height = height * stride;
	for (int c = 0; c < nchannels; ++c) {
		for (int y = 0; y < us_height; ++y) {
			for (int x = 0; x < us_width; ++x) {
				int y0 = y / stride;
				int x0 = x / stride;
				float val = in[c * width * height + y0 * width + x0];
				out[c * us_width * us_height + y * us_width + x] = val;
			}
		}
	}
}