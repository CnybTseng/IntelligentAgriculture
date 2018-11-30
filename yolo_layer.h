#ifndef _YOLO_LAYER_H_
#define _YOLO_LAYER_H_

#include "znet.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int batch_size;
	int ninputs;
	int noutputs;
	int nscales;
	int total_scales;
	int classes;
	int *mask;
	int *anchor_boxes;
	float *input;
	float *output;
} yolo_layer;

void free_yolo_layer(void *_layer);
void print_yolo_layer_info(void *_layer, int id);
void set_yolo_layer_input(void *_layer, float *input);
float *get_yolo_layer_output(void *_layer);
void forward_yolo_layer(void *_layer, znet *net);
void get_yolo_layer_detections(yolo_layer *layer, znet *net, int imgw, int imgh, float thresh, list *l);
void free_yolo_layer_detections(list *l);
void backward_yolo_layer(yolo_layer *layer, znet *net);

#ifdef __cplusplus
}
#endif

#endif