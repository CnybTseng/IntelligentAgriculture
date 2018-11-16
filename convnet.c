#include <stdio.h>
#include <stdlib.h>
#include "convnet.h"

convnet *convnet_create(void *layers[], int nlayers)
{
	convnet *net = (convnet *)malloc(sizeof(convnet));
	if (!net) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return net;
	}
	
	net->work_mode = INFERENCE;
	net->nlayers = nlayers;
	net->layers = layers;
	
	convolutional_layer *layer = (convolutional_layer *)layers[0];
	layer->input = net->input;
	
	return net;
}

void convnet_train(convnet *net, datastore *ds, train_options *options)
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
			
		} else if (type == UPSAMPLE) {
			
		} else if (type == YOLO) {
			
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
	
	free(net);
	net = NULL;
}

void convnet_architecture(convnet *net)
{
	printf("id\tlayer\t\t\tinput\t\tsize/stride\t\tfilters\t\t\toutput\n");
	
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
			
		} else if (type == UPSAMPLE) {
			
		} else if (type == YOLO) {
			
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}