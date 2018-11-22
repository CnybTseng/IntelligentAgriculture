#ifndef _CONVNET_H_
#define _CONVNET_H_

#include <stdio.h>
#include <stdlib.h>
#include "list.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	INFERENCE, TRAIN
} WORK_MODE;

typedef enum {
	CONVOLUTIONAL, MAXPOOL, ROUTE, RESAMPLE, YOLO
} LAYER_TYPE;

typedef enum {
	RELU, LEAKY, LINEAR, LOGISTIC
} ACTIVATION;

typedef struct {
	
} datastore;

typedef struct {
	
} train_options;

struct convnet;
typedef struct convnet convnet;

typedef void (*PRINT_LAYER_INFO)(void *layer, int id);
typedef void (*SET_LAYER_INPUT)(void *layer, float *input);
typedef float *(*GET_LAYER_OUTPUT)(void *layer);
typedef void (*FORWARD)(void *layer, convnet *net);
typedef void (*FREE_LAYER)(void *layer);

struct convnet {
	WORK_MODE work_mode;
	int nlayers;
	void **layers;
	float *input;
	float *output;
	int width;
	int height;
	PRINT_LAYER_INFO *print_layer_info;
	SET_LAYER_INPUT *set_layer_input;
	GET_LAYER_OUTPUT *get_layer_output;
	FORWARD *forward;
	FREE_LAYER *free_layer;
	int *is_output_layer;
};

typedef struct {
	int w, h, c;
} dim3;

typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;

typedef struct {
	float x, y, w, h;
} box;

typedef struct {
	box bbox;
	int classes;
	float *probabilities;
	float objectness;
} detection;

/** @name 卷积网络层的创建.目前仅支持卷积层,最大池化层,重采样层,路线层和瞄一眼层.
 ** @ { */
void *make_convolutional_layer(ACTIVATION activation, dim3 input_size, int filter_size, int nfilters,
                               int stride, int padding, int batch_size, int batch_norm, dim3 *output_size);
void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size);
void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask,
                      int *anchor_boxes);
void *make_route_layer(int batch_size, int *input_layers, int *input_sizes, int nroutes);
void *make_resample_layer(dim3 input_size, int batch_size, int stride, dim3 *output_size);
/** @ }*/

/** @name 卷积网络的创建,训练,推断,销毁等操作.
 ** @ { */
convnet *convnet_create(void *layers[], int nlayers);
void convnet_train(convnet *net, datastore *ds, train_options *opts);
float *convnet_inference(convnet *net, image *input);
void convnet_destroy(convnet *net);
void convnet_architecture(convnet *net);
/** @ } */

list *get_detections(convnet *net, float thresh, int width, int height);
void free_detections(list *detections);

#ifdef __cplusplus
}
#endif

#endif