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
	layer->biases = NULL;
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

void free_yolo_layer(void *_layer)
{
	yolo_layer *layer = (yolo_layer *)_layer;
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

void print_yolo_layer_info(void *_layer, int id)
{
	printf("%02d\tyolo\n", id);
}

void set_yolo_layer_input(void *_layer, float *input)
{
	yolo_layer *layer = (yolo_layer *)_layer;
	layer->input = input;
}

float *get_yolo_layer_output(void *_layer)
{
	yolo_layer *layer = (yolo_layer *)_layer;
	return layer->output;
}

void forward_yolo_layer(void *_layer, convnet *net)
{
	yolo_layer *layer = (yolo_layer *)_layer;
	int total = layer->ninputs * layer->batch_size;
	memcpy(layer->output, layer->input, total * sizeof(float));
	
	int volume_per_scale = layer->output_size.w * layer->output_size.h * (4 + 1 + layer->classes);
	if (net->work_mode == INFERENCE) {
		for (int b = 0; b < layer->batch_size; ++b) {
			for (int s = 0; s < layer->nscales; ++s) {
				float *at = layer->output + b * layer->noutputs + s * volume_per_scale;
				activate(at, 2 * layer->output_size.w * layer->output_size.h, LOGISTIC);
				at += 4 * layer->output_size.w * layer->output_size.h;
				activate(at, (1 + layer->classes) * layer->output_size.w * layer->output_size.h, LOGISTIC);
			}
		}
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
	
	static int timer = 0;
	char filename[128];
	sprintf(filename, "yolo_%d.txt", timer++);
	save_yolo_layer_output(layer, filename);
}

int get_yolo_layer_detect_num(yolo_layer *layer, float thresh)
{
	int num = 0;
	int size = layer->output_size.w * layer->output_size.h;
	int volume_per_scale = size * (4 + 1 + layer->classes);
	for (int s = 0; s < layer->nscales; ++s) {
		float *at = layer->output + s * volume_per_scale + 4 * size;
		for (int i = 0; i < size; ++i) {
			if (at[i] > thresh) ++num;
		}
	}
	
	return num;
}

void backward_yolo_layer(yolo_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void save_yolo_layer_output(yolo_layer *layer, const char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	for (int c = 0; c < layer->output_size.c; ++c) {
		fprintf(fp, "channel=%d\n", c);
		float *at = layer->output + c * layer->output_size.w * layer->output_size.h;
		for (int y = 0; y < layer->output_size.h; ++y) {
			for (int x = 0; x < layer->output_size.w; ++x) {
				fprintf(fp, "%.7f ", at[y * layer->output_size.w + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n", fp);
	}
	
	fclose(fp);
}