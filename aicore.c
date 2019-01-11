/** @file aicore.c - Implementation
 ** @brief 智慧农业核心模块
 ** @author 曾志伟
 ** @date 2018.11.16
 **/

/*
Copyright (C) 2018 Chengdu ZLT Technology Co., Ltd.
All rights reserved.

This file is part of the smart agriculture toolkit and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include "aicore.h"
#include "znet.h"
#include "list.h"
#include "fifo.h"

#define CACHE_IMAGE_NUMBER							8
#define NETWORK_INPUT_WIDTH							416
#define NETWORK_INPUT_HEIGHT						416

typedef enum {
	THREAD_RUNNING,
	THREAD_DEAD
} THREAD_STATUS;

static void *layers[24];
static znet *net = NULL;
static pthread_once_t module_create_once_control = PTHREAD_ONCE_INIT;
static int init_status;
static pthread_t image_enqueue_tid;
static pthread_t image_process_tid;
static pthread_t image_dequeue_tid;
static THREAD_STATUS thread_status;
static ai_core_param core_param;
static Fifo *raw_image_queue = NULL;
static Fifo *normalized_image_queue = NULL;
static Fifo *output_image_queue = NULL;
static char *image_input_queue_input_buffer = NULL;
static char *image_input_queue_output_buffer = NULL;
static char *image_output_queue_input_buffer = NULL;
static char *image_output_queue_output_buffer = NULL;

static void ai_core_init_routine();
static void create_image_fifo();
static int create_image_enqueue_thread();
static int create_image_process_thread();
static int create_image_dequeue_thread();
static void *image_enqueue_thread(void *param);
static void *image_process_thread(void *param);
static void *image_dequeue_thread(void *param);
static void wait_for_thread_dead(pthread_t tid);
static unsigned int roundup_power_of_2(unsigned int a);

int ai_core_init(void *param)
{
	if (param) {
		core_param.image_width = ((ai_core_param *)param)->image_width;
		core_param.image_height = ((ai_core_param *)param)->image_height;
	} else {
		core_param.image_width = 1920;
		core_param.image_height = 1080;
	}
	
	pthread_once(&module_create_once_control, &ai_core_init_routine);
	
	return init_status;
}

int ai_core_push_image(const char *bmp, size_t size)
{
	printf("image size %d\n", size);
	return AIC_OK;
}

size_t ai_core_pull_image(char *bmp)
{
	return 0;
}

void ai_core_free()
{
	thread_status = THREAD_DEAD;
	wait_for_thread_dead(image_enqueue_tid);
	wait_for_thread_dead(image_process_tid);
	wait_for_thread_dead(image_dequeue_tid);
	znet_destroy(net);
	fifo_delete(raw_image_queue);
	fifo_delete(normalized_image_queue);
	fifo_delete(output_image_queue);
}

int create_image_enqueue_thread()
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(&image_enqueue_tid, &attr, image_enqueue_thread, NULL);
	if (0 != ret) {
		fprintf(stderr, "pthread_create[%s:%d].\n", __FILE__, __LINE__);
		thread_status = THREAD_DEAD;
		return -1;
	}
	
	return 0;
}

int create_image_process_thread()
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(&image_process_tid, &attr, image_process_thread, NULL);
	if (0 != ret) {
		fprintf(stderr, "pthread_create[%s:%d].\n", __FILE__, __LINE__);
		thread_status = THREAD_DEAD;
		return -1;
	}
	
	return 0;
}

int create_image_dequeue_thread()
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(&image_dequeue_tid, &attr, image_dequeue_thread, NULL);
	if (0 != ret) {
		fprintf(stderr, "pthread_create[%s:%d].\n", __FILE__, __LINE__);
		thread_status = THREAD_DEAD;
		return -1;
	}
		
	return 0;
}

void ai_core_init_routine()
{
	int nlayers = sizeof(layers) / sizeof(layers[0]);
	dim3 output_size;
	
	int bigger_mask[] = {3, 4, 5};
	int smaller_mask[] = {0, 1, 2};
	int anchor_boxes[] = {61,117, 62,191, 199,118, 128,195, 92,293, 191,291};
	const int scales = 3;
	const int classes = 1;
	const int object_tensor_depth = (4 + 1 + classes) * scales;
	
	dim3 input_size = {NETWORK_INPUT_WIDTH, NETWORK_INPUT_HEIGHT, 3};
	layers[0] = make_convolutional_layer(LEAKY, input_size, 3, 16, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[1] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[2] = make_convolutional_layer(LEAKY, input_size, 3, 32, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[3] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[4] = make_convolutional_layer(LEAKY, input_size, 3, 64, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[5] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[6] = make_convolutional_layer(LEAKY, input_size, 3, 128, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[7] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[8] = make_convolutional_layer(LEAKY, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[9] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[10] = make_convolutional_layer(LEAKY, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[11] = make_maxpool_layer(input_size, 2, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[12] = make_convolutional_layer(LEAKY, input_size, 3, 1024, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[13] = make_convolutional_layer(LEAKY, input_size, 1, 256, 1, 0, 1, 1, &output_size);
	input_size = output_size;
	layers[14] = make_convolutional_layer(LEAKY, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[15] = make_convolutional_layer(LINEAR, input_size, 1, object_tensor_depth, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	layers[16] = make_yolo_layer(input_size, 1, 3, 6, classes, bigger_mask, anchor_boxes);
	
	int layer_ids1[] = {13};
	void *routes1[] = {layers[13]};
	layers[17] = make_route_layer(1, 1, routes1, layer_ids1, &output_size);
	
	input_size = output_size;
	layers[18] = make_convolutional_layer(LEAKY, input_size, 1, 128, 1, 0, 1, 1, &output_size);
	
	input_size = output_size;
	layers[19] = make_resample_layer(input_size, 1, 2, &output_size);
	
	int layer_ids2[] = {19, 8};
	void *routes2[] = {layers[19], layers[8]};
	layers[20] = make_route_layer(1, 2, routes2, layer_ids2, &output_size);
	
	input_size = output_size;
	layers[21] = make_convolutional_layer(LEAKY, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[22] = make_convolutional_layer(LINEAR, input_size, 1, object_tensor_depth, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	layers[23] = make_yolo_layer(input_size, 1, 3, 6, classes, smaller_mask, anchor_boxes);

	net = znet_create(layers, nlayers, "agriculture.weights");
	if (!net) {
		init_status = AIC_INIT_FAIL;
		return;
	}
	
	raw_image_queue = fifo_alloc(roundup_power_of_2(core_param.image_width * core_param.image_height * 3 * CACHE_IMAGE_NUMBER));
	if (!raw_image_queue) {
		init_status = AIC_ALLOCATE_FAIL;
		return;
	}
	
	normalized_image_queue = fifo_alloc(roundup_power_of_2(NETWORK_INPUT_WIDTH * NETWORK_INPUT_HEIGHT * 3 * CACHE_IMAGE_NUMBER));
	if (!normalized_image_queue) {
		init_status = AIC_ALLOCATE_FAIL;
		return;
	}
	
	output_image_queue = fifo_alloc(roundup_power_of_2(core_param.image_width * core_param.image_height * 3 * CACHE_IMAGE_NUMBER));
	if (!output_image_queue) {
		init_status = AIC_ALLOCATE_FAIL;
		return;
	}
	
	thread_status = THREAD_RUNNING;
	if (create_image_enqueue_thread() || create_image_process_thread() || create_image_dequeue_thread()) {
		init_status = AIC_INIT_FAIL;
		return;
	}
	
	init_status = AIC_OK;
	znet_architecture(net);
}

void *image_enqueue_thread(void *param)
{
	struct timespec req = {0, 100000000};
	while (thread_status == THREAD_RUNNING) {
		nanosleep(&req, NULL);
	}
	
	return 0;
}

void *image_process_thread(void *param)
{
	struct timespec req = {0, 100000000};
	while (thread_status == THREAD_RUNNING) {
		nanosleep(&req, NULL);
	}
	
	return 0;
}

void *image_dequeue_thread(void *param)
{
	struct timespec req = {0, 100000000};
	while (thread_status == THREAD_RUNNING) {
		nanosleep(&req, NULL);
	}
	
	return 0;
}

void wait_for_thread_dead(pthread_t tid)
{
	int timer = 1000;
	while (timer--) {
		int ret = pthread_kill(tid, 0);
		if (ESRCH == ret) {
			fprintf(stderr, "the thread didn't exists or already quit[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else if (EINVAL == ret) {
			fprintf(stderr, "signal is invalid[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else {
			continue;
		}
	}
}

unsigned int roundup_power_of_2(unsigned int a)
{
	unsigned int position;
	int i;
	
	if (a == 0) {
		return 0;
	}

	position = 0;
	for (i = a; i != 0; i >>= 1) {
		position++;
	}

	return (unsigned int)(1 << position);
}