#ifndef _YOLO_LAYER_H_
#define _YOLO_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void print_yolo_layer_info(void *_layer, int id);
void set_yolo_layer_input(void *_layer, float *input);
float *get_yolo_layer_output(void *_layer);
void forward_yolo_layer(void *_layer, convnet *net);
int get_yolo_layer_detect_num(yolo_layer *layer, float thresh);
void backward_yolo_layer(yolo_layer *layer, convnet *net);
void save_yolo_layer_output(yolo_layer *layer, const char *filename);

#ifdef __cplusplus
}
#endif

#endif