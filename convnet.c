#include <stdio.h>
#include <stdlib.h>
#include "convnet.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "route_layer.h"
#include "resample_layer.h"
#include "yolo_layer.h"

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
	
	FILE *fp = fopen("yolov3-tiny.weights", "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		convnet_destroy(net);
		return net;
	}
	
	int major;
	int minor;
	int revision;
	unsigned long long seen;
	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	fread(&seen, sizeof(unsigned long long), 1, fp);
	printf("version %d.%d.%d, seen %u.\n", major, minor, revision, (unsigned int)seen);
	
	for (int i = 0; i < nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			load_convolutional_layer_weights(layer, fp);
		}
	}
	
	fclose(fp);
	
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
	net->input = input->data;
	
	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			printf("%d conv\n", i + 1);
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			layer->input = net->input;
			forward_convolutional_layer(layer, net);
			net->input = layer->output;
		} else if (type == MAXPOOL) {
			printf("%d maxpool\n", i + 1);
			maxpool_layer *layer = (maxpool_layer *)net->layers[i];
			layer->input = net->input;
			forward_maxpool_layer(layer, net);
			net->input = layer->output;
		} else if (type == ROUTE) {
			printf("%d route\n", i + 1);
			route_layer *layer = (route_layer *)net->layers[i];
			forward_route_layer(layer, net);
			net->input = layer->output;
		} else if (type == RESAMPLE) {
			printf("%d resample\n", i + 1);
			resample_layer *layer = (resample_layer *)net->layers[i];
			layer->input = net->input;
			forward_resample_layer(layer, net);
		} else if (type == YOLO) {
			printf("%d yolo\n", i + 1);
			yolo_layer *layer = (yolo_layer *)net->layers[i];
			layer->input = net->input;
			forward_yolo_layer(layer, net);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}

	return 0;
}

void convnet_destroy(convnet *net)
{
	if (!net) return;
	
	for (int i = 0; i < net->nlayers; ++i) {
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