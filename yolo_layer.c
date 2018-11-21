#include <math.h>
#include <string.h>
#include "yolo_layer.h"
#include "activation.h"
#include "zutils.h"

static box get_yolo_box(float *box_volume, int id, int layer_width, int layer_height,
                        int net_width, int net_height, int *anchor_box,
						int image_width, int image_height);
static float *get_yolo_prob(float *prob_volume, int id, int layer_width, int layer_height,
                            int classes, float objectness);

void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask,
                      int *anchor_boxes)
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
	layer->anchor_boxes = NULL;
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
	
	layer->anchor_boxes = calloc(total_scales * 2, sizeof(int));
	if (!layer->anchor_boxes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
		
	for (int i = 0; i < total_scales; ++i) {
		layer->anchor_boxes[2 * i] = anchor_boxes[2 * i];
		layer->anchor_boxes[2 * i + 1] = anchor_boxes[2 * i + 1];
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
	
	if (layer->anchor_boxes) {
		free(layer->anchor_boxes);
		layer->anchor_boxes = NULL;
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
}

void get_yolo_layer_detections(yolo_layer *layer, convnet *net, int imgw, int imgh, float thresh)
{
	int counter = 1;
	int size = layer->output_size.w * layer->output_size.h;
	int volume_per_scale = size * (4 + 1 + layer->classes);
	for (int s = 0; s < layer->nscales; ++s) {
		float *box_vol = layer->output + s * volume_per_scale;
		float *obj_slc = layer->output + s * volume_per_scale + 4 * size;
		float *prob_vol = layer->output + s * volume_per_scale + 5 * size;
		for (int i = 0; i < size; ++i) {
			if (obj_slc[i] < thresh) continue;
			detection det;
			det.bbox = get_yolo_box(box_vol, i, layer->output_size.w, layer->output_size.h,
				net->width, net->height, &layer->anchor_boxes[2 * layer->mask[s]], imgw, imgh);
			det.classes = layer->classes;
			det.probabilities = get_yolo_prob(prob_vol, i, layer->output_size.w, layer->output_size.h,
				layer->classes, obj_slc[i]);
			det.objectness = obj_slc[i];
			
			printf("%d bbox[%f,%f,%f,%f] classes[%d] objectness[%f] probabilities[",
				counter, det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h, det.classes, det.objectness);
			for (int j = 0; j < det.classes; ++j) {
				printf("%.1f ", det.probabilities[j]);
			}
			
			printf("\n\n");
			counter++;
			free(det.probabilities);
		}
	}
}

void backward_yolo_layer(yolo_layer *layer, convnet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

box get_yolo_box(float *box_volume, int id, int layer_width, int layer_height,
                 int net_width, int net_height, int *anchor_box,
				 int image_width, int image_height)
{
	box b;
	int slice_size = layer_width * layer_height;
	b.x = (id % layer_width + box_volume[id]) / layer_width;
	b.y = (id / layer_width + box_volume[id + slice_size]) / layer_height;
	b.w = exp(box_volume[id + 2 * slice_size]) * anchor_box[0] / net_width;
	b.h = exp(box_volume[id + 3 * slice_size]) * anchor_box[1] / net_height;
	
	float sx = net_width / (float)image_width;
	float sy = net_height / (float)image_height;
	float s = sx < sy ? sx : sy;
	
	int rsz_width = (int)(s * image_width);
	int rsz_height = (int)(s * image_height);

	b.x = (b.x - (net_width - rsz_width) / 2.0f / net_width) * net_width / rsz_width;
	b.y = (b.y - (net_height - rsz_height) / 2.0f / net_height) * net_height / rsz_height;
	b.w = b.w * net_width / rsz_width;
	b.h = b.h * net_height / rsz_height;
	
	return b;
}

float *get_yolo_prob(float *prob_volume, int id, int layer_width, int layer_height,
                     int classes, float objectness)
{
	float *probabilities = calloc(classes, sizeof(float));
	if (!probabilities) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return probabilities;
	}
	
	int slice_size = layer_width * layer_height;
	for (int i = 0; i < classes; ++i) {
		probabilities[i] = objectness * prob_volume[id + slice_size * i];
	}
	
	return probabilities;
}