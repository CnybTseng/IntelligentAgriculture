#include <stdio.h>
#include <stdlib.h>
#include "convnet.h"

convnet *convnet_create(void *layers[], int nlayers)
{
	convnet *net = calloc(1, sizeof(convnet));
	if (!net) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return net;
	}
	
	net->work_mode = INFERENCE;
	net->nlayers = nlayers;
	net->layers = layers;
	net->input = NULL;
	net->output = NULL;
	
	convolutional_layer *layer = (convolutional_layer *)layers[0];
	layer->input = net->input;
	
	return net;
}

void convnet_train(convnet *net, datastore *ds, train_options *opts)
{
	net->work_mode = TRAIN;
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

float *convnet_inference(convnet *net, image *input)
{
	net->work_mode = INFERENCE;

	return 0;
}

void convnet_destroy(convnet *net)
{
	if (!net) return;
	
	for (int i = 0; i < net->nlayers; i++) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			free_convolution_layer(layer);
		} else if (type == MAXPOOL) {
			maxpool_layer *layer = (maxpool_layer *)net->layers[i];
			free_maxpool_layer(layer);
		} else if (type == ROUTE) {
			route_layer *layer = (route_layer *)net->layers[i];
			free_route_layer(layer);
		} else if (type == RESAMPLE) {
			resample_layer *layer = (resample_layer *)net->layers[i];
			free_resample_layer(layer);
		} else if (type == YOLO) {
			yolo_layer *layer = (yolo_layer *)net->layers[i];
			free_yolo_layer(layer);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
	
	free(net);
	net = NULL;
}

void convnet_architecture(convnet *net)
{
	printf("id\tlayer\t\t\t   input\t  size/stride\t     filters\t\t\t  output\n");
	
	for (int i = 0; i < net->nlayers; i++) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			printf("%2d\tconv\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\n",
				i + 1,
				layer->input_size.w,
				layer->input_size.h,
				layer->input_size.c,
				layer->filter_size,
				layer->filter_size,
				layer->stride,
				layer->nfilters,
				layer->output_size.w,
				layer->output_size.h,
				layer->output_size.c);
		} else if (type == MAXPOOL) {
			maxpool_layer *layer = (maxpool_layer *)net->layers[i];
			printf("%2d\tmaxpool\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\n",
				i + 1,
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
		} else if (type == ROUTE) {
			route_layer *layer = (route_layer *)net->layers[i];
			printf("%02d\troute ", i + 1);
			for (int i = 0; i < layer->nroutes; ++i) {
				printf("%d", layer->input_layers[i] + 1);
				if (i < layer->nroutes - 1) printf(",");
			}
			printf("\n");
		} else if (type == RESAMPLE) {
			resample_layer *layer = (resample_layer *)net->layers[i];
			printf("%2d\tresample\t%4d x%4d x%4d\t\t%d\t\t\t\t%4d x%4d x%4d\n",
				i + 1,
				layer->input_size.w,
				layer->input_size.h,
				layer->input_size.c,
				layer->stride,
				layer->output_size.w,
				layer->output_size.h,
				layer->output_size.c);
		} else if (type == YOLO) {
			printf("%02d\tyolo\n", i + 1);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}