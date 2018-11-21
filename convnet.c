#include <stdio.h>
#include <stdlib.h>
#include "convnet.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "route_layer.h"
#include "resample_layer.h"
#include "yolo_layer.h"

static int convnet_parse_input_size(convnet *net);
static int convnet_parse_layer(convnet *net);
static int convnet_parse_weights(convnet *net);

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
	net->print_layer_info = NULL;
	net->set_layer_input = NULL;
	net->get_layer_output = NULL;
	net->forward = NULL;
	net->free_layer = NULL;
	net->is_output_layer = NULL;
	
	net->print_layer_info = calloc(nlayers, sizeof(PRINT_LAYER_INFO));
	if (!net->print_layer_info) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->set_layer_input = calloc(nlayers, sizeof(SET_LAYER_INPUT));
	if (!net->set_layer_input) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->get_layer_output = calloc(nlayers, sizeof(GET_LAYER_OUTPUT));
	if (!net->get_layer_output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->forward = calloc(nlayers, sizeof(FORWARD));
	if (!net->forward) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->free_layer = calloc(nlayers, sizeof(FREE_LAYER));
	if (!net->free_layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->is_output_layer = calloc(nlayers, sizeof(int));
	if (!net->is_output_layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	if (convnet_parse_input_size(net)) goto cleanup;
	if (convnet_parse_layer(net)) goto cleanup;
	
	if (convnet_parse_weights(net)) {
		cleanup:convnet_destroy(net);
		return NULL;
	}
	
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
		net->set_layer_input[i](net->layers[i], net->input);
		net->forward[i](net->layers[i], net);
		net->input = net->get_layer_output[i](net->layers[i]);
	}

	return 0;
}

void convnet_destroy(convnet *net)
{
	if (!net) return;
	
	for (int i = 0; i < net->nlayers; ++i) {		
		net->free_layer[i](net->layers[i]);
	}
	
	if (net->print_layer_info) {
		free(net->print_layer_info);
		net->print_layer_info = NULL;
	}
	
	if (net->set_layer_input) {
		free(net->set_layer_input);
		net->set_layer_input = NULL;
	}
	
	if (net->get_layer_output) {
		free(net->get_layer_output);
		net->get_layer_output = NULL;
	}
	
	if (net->forward) {
		free(net->forward);
		net->forward = NULL;
	}
	
	if (net->free_layer) {
		free(net->free_layer);
		net->free_layer = NULL;
	}
	
	if (net->is_output_layer) {
		free(net->is_output_layer);
		net->is_output_layer = NULL;
	}

	free(net);
	net = NULL;
}

void convnet_architecture(convnet *net)
{
	printf("id\tlayer\t\t\t   input\t  size/stride\t     filters\t\t\t  output\n");
	for (int i = 0; i < net->nlayers; i++) {
		net->print_layer_info[i](net->layers[i], i + 1);
	}
}

detection *get_detections(convnet *net, float thresh, int width, int height, int *ndets)
{
	for (int i = 0; i < net->nlayers; ++i) {
		if (!net->is_output_layer[i]) continue;
		get_yolo_layer_detections((yolo_layer *)net->layers[i], net, width, height, thresh);
	}
	
	return NULL;
}

int convnet_parse_input_size(convnet *net)
{
	LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[0]);
	if (type != CONVOLUTIONAL) {
		fprintf(stderr, "the first layer isn't convolutional layer!");
		return -1;
	}
	
	convolutional_layer *layer = (convolutional_layer *)net->layers[0];
	net->width = layer->input_size.w;
	net->height = layer->input_size.h;
	
	return 0;
}

int convnet_parse_layer(convnet *net)
{
	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			net->print_layer_info[i] = print_convolutional_layer_info;
			net->set_layer_input[i] = set_convolutional_layer_input;
			net->get_layer_output[i] = get_convolutional_layer_output;
			net->forward[i] = forward_convolutional_layer;
			net->free_layer[i] = free_convolution_layer;
		} else if (type == MAXPOOL) {
			net->print_layer_info[i] = print_maxpool_layer_info;
			net->set_layer_input[i] = set_maxpool_layer_input;
			net->get_layer_output[i] = get_maxpool_layer_output;
			net->forward[i] = forward_maxpool_layer;
			net->free_layer[i] = free_maxpool_layer;
		} else if (type == ROUTE) {
			net->print_layer_info[i] = print_route_layer_info;
			net->set_layer_input[i] = set_route_layer_input;
			net->get_layer_output[i] = get_route_layer_output;
			net->forward[i] = forward_route_layer;
			net->free_layer[i] = free_route_layer;
		} else if (type == RESAMPLE) {
			net->print_layer_info[i] = print_resample_layer_info;
			net->set_layer_input[i] = set_resample_layer_input;
			net->get_layer_output[i] = get_resample_layer_output;
			net->forward[i] = forward_resample_layer;
			net->free_layer[i] = free_resample_layer;
		} else if (type == YOLO) {
			net->print_layer_info[i] = print_yolo_layer_info;
			net->set_layer_input[i] = set_yolo_layer_input;
			net->get_layer_output[i] = get_yolo_layer_output;
			net->forward[i] = forward_yolo_layer;
			net->free_layer[i] = free_yolo_layer;
			net->is_output_layer[i] = 1;
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
			return -1;
		}
	}
	
	return 0;
}

int convnet_parse_weights(convnet *net)
{
	FILE *fp = fopen("yolov3-tiny.weights", "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return -1;
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
	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			load_convolutional_layer_weights(layer, fp);
		}
	}
	
	fclose(fp);
	
	return 0;
}