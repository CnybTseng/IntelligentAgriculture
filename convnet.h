#ifndef _CONVNET_H_
#define _CONVNET_H_

#include <stdio.h>
#include <stdlib.h>

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
	RELU, LINEAR, LOGISTIC
} ACTIVATION;

typedef struct {
	
} datastore;

typedef struct {
	
} train_options;

struct convnet;
typedef struct convnet convnet;

struct convnet {
	WORK_MODE work_mode;
	int nlayers;
	void **layers;
	float *input;
	float *output;
};

typedef struct {
	int w, h, c;
} dim3;

typedef struct {
	LAYER_TYPE type;
	ACTIVATION activation;
	dim3 input_size;
	dim3 output_size;
	int filter_size;
	int nfilters;
	int stride;
	int padding;
	int batch_size;
	int batch_norm;
	int nweights;
	int nbiases;
	int ninputs;
	int vmsize;
	int noutputs;
	float *weights;
	float *scales;
	float *biases;
	float *rolling_mean;
	float *rolling_variance;
	float *input;
	float *vecmat;
	float *output;
	void (*forward)(void *layer, convnet *net);
	void (*backward)(void *layer, convnet *net);
	void (*destroy)(void *layer);
} convolutional_layer;

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int filter_size;
	int stride;
	int padding;
	int batch_size;
	int ninputs;
	int noutputs;
	float *input;
	float *output;
	void (*forward)(void *layer, convnet *net);
	void (*backward)(void *layer, convnet *net);
	void (*destroy)(void *layer);
} maxpool_layer;

typedef struct {
	LAYER_TYPE type;
	int batch_size;
	int ninputs;
	int noutputs;
	int nroutes;
	int *input_layers;
	int *input_sizes;
	float *output;
} route_layer;

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int stride;
	int batch_size;
	int ninputs;
	int noutputs;
	int upsample;
	float *input;
	float *output;
} resample_layer;

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
	float *input;
	float *output;
} yolo_layer;

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
	float *probability;
	float objectness;
} detection;

/** @name 神经网络层的创建和销毁.
 ** @ { */
void *make_convolutional_layer(ACTIVATION activation, dim3 input_size, int filter_size, int nfilters,
                               int stride, int padding, int batch_size, int batch_norm, dim3 *output_size);
void free_convolution_layer(convolutional_layer *layer);
void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size);
void free_maxpool_layer(maxpool_layer *layer);
void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask);
void free_yolo_layer(yolo_layer *layer);
void *make_route_layer(int batch_size, int *input_layers, int *input_sizes, int nroutes);
void free_route_layer(route_layer *layer);
void *make_resample_layer(dim3 input_size, int batch_size, int stride, dim3 *output_size);
void free_resample_layer(resample_layer *layer);
/** @ }*/

/** @name 卷积神经网络的创建, 训练, 推断, 销毁等操作.
 ** @ { */
convnet *convnet_create(void *layers[], int nlayers);
void convnet_train(convnet *net, datastore *ds, train_options *opts);
float *convnet_inference(convnet *net, image *input);
void convnet_destroy(convnet *net);
void convnet_architecture(convnet *net);
/** @ } */

#ifdef __cplusplus
}
#endif

#endif