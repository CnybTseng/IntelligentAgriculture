#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "convnet.h"
#include "im2col.h"
#include "zutils.h"
#include "gemm.h"
#include "activation.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "bmp.h"
#include "image.h"

void test_multi_free(int argc, char *argv[]);
void test_convnet(int argc, char *argv[]);
void test_im2col(int argc, char *argv[]);
void test_gemm(int argc, char *argv[]);
void test_activate(int argc, char *argv[]);
void test_convolutional_layer(int argc, char *argv[]);
void test_maxpool_layer(int argc, char *argv[]);
void test_mset(int argc, char *argv[]);
void test_mcopy(int argc, char *argv[]);
void test_bmp(int argc, char *argv[]);
void test_split(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	test_split(argc, argv);
	
	return 0;
}

void test_multi_free(int argc, char *argv[])
{
	char *buf1 = (char *)malloc(1024);
	if (!buf1) {
		return;
	}
	
	int *buf2 = (int *)malloc(1024);
	if (!buf2) {
		return mmfree(1, buf1);
	}
	
	float *buf3 = (float *)malloc(1024);
	if (!buf3) {
		return mmfree(2, buf1, buf2);
	}
	
	mmfree(3, buf1, buf2, buf3);
}

void test_convnet(int argc, char *argv[])
{	
	void *layers[24];
	int nlayers = sizeof(layers) / sizeof(layers[0]);
	dim3 output_size;
	
	dim3 input_size = {416, 416, 3};
	layers[0] = make_convolutional_layer(RELU, input_size, 3, 16, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[1] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[2] = make_convolutional_layer(RELU, input_size, 3, 32, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[3] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[4] = make_convolutional_layer(RELU, input_size, 3, 64, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[5] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[6] = make_convolutional_layer(RELU, input_size, 3, 128, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[7] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[8] = make_convolutional_layer(RELU, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[9] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[10] = make_convolutional_layer(RELU, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[11] = make_maxpool_layer(input_size, 2, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[12] = make_convolutional_layer(RELU, input_size, 3, 1024, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[13] = make_convolutional_layer(RELU, input_size, 1, 256, 1, 0, 1, 1, &output_size);
	input_size = output_size;
	layers[14] = make_convolutional_layer(RELU, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[15] = make_convolutional_layer(LINEAR, input_size, 1, 255, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	int bigger_mask[] = {3, 4, 5};
	layers[16] = make_yolo_layer(input_size, 1, 3, 6, 80, bigger_mask);
	
	int input_layers[] = {13};
	convolutional_layer *layer = (convolutional_layer *)layers[13];
	int input_sizes[] = {layer->noutputs};
	layers[17] = make_route_layer(1, input_layers, input_sizes, 1);
	
	input_size = layer->output_size;
	layers[18] = make_convolutional_layer(RELU, input_size, 1, 128, 1, 0, 1, 1, &output_size);
	
	input_size = output_size;
	layers[19] = make_resample_layer(input_size, 1, 2, &output_size);
	
	int route_layers[] = {19, 8};
	int route_sizes[2];
	resample_layer *rsl = (resample_layer *)layers[route_layers[0]];
	route_sizes[0] = rsl->noutputs;
	layer = (convolutional_layer *)layers[route_layers[1]];
	route_sizes[1] = layer->noutputs;
	layers[20] = make_route_layer(1, route_layers, route_sizes, 2);
	
	layer = (convolutional_layer *)layers[route_layers[1]];
	input_size.w = layer->output_size.w;
	input_size.h = layer->output_size.w;
	input_size.c = layer->output_size.c;
	rsl = (resample_layer *)layers[route_layers[0]];
	input_size.c += rsl->output_size.c;
	layers[21] = make_convolutional_layer(RELU, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[22] = make_convolutional_layer(LINEAR, input_size, 1, 255, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	int smaller_mask[] = {0, 1, 2};
	layers[23] = make_yolo_layer(input_size, 1, 3, 6, 80, smaller_mask);
	
	convnet *net = convnet_create(layers, nlayers);
	convnet_architecture(net);
	
	image input = {416, 416, 3, NULL};
	input.data = calloc(input.w * input.h * input.c, sizeof(float));
	if (!input.data) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
	}
	
	srand(time(NULL));
	for (int i = 0; i < input.w * input.h * input.c; ++i) {
		input.data[i] = rand() / (double)RAND_MAX;
	}
	
	convnet_inference(net, &input);
	
	free(input.data);
	convnet_destroy(net);
}

void test_im2col(int argc, char *argv[])
{
	int width = 8;
	int height = 8;
	int nchannels = 3;
	int fsize = 3;
	int stride = 2;
	int padding = 1;
	
	float *image = (float *)malloc(width * height * nchannels * sizeof(float));
	if (!image) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		exit(-1);
	}
	
	int convw = (width + 2 * padding - fsize) / stride + 1;
	int convh = (height + 2 * padding - fsize) / stride + 1;
	
	float *matrix = (float *)malloc(fsize * fsize * nchannels * convw * convh * sizeof(float));
	if (!matrix) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(1, image);
		exit(-1);
	}
		
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < width * height; i++) {
			image[i + c * width * height] = 1 + i + c * width * height;
		}
	}
	
	im2col_cpu(image, width, height, nchannels, fsize, stride, padding, matrix);
	
	FILE *fp = fopen("matrix.txt", "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		mmfree(2, image, matrix);
		exit(-1);
	}
	
	for (int y = 0; y < fsize * fsize * nchannels; y++) {
		for (int x = 0; x < convw * convh; x++) {
			fprintf(fp, "%.0f\t", matrix[y * convw * convh + x]);
		}
		fputs("\n", fp);
	}
	
	fclose(fp);
	mmfree(2, image, matrix);
}

void test_gemm(int argc, char *argv[])
{
	int aw = 16;
	int ah = 8;
	int bw = 8;
	int bh = 16;
	
	float *A = (float *)malloc(aw * ah * sizeof(float));
	if (!A) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		exit(-1);
	}
	
	float *B = (float *)malloc(bw * bh * sizeof(float));
	if (!B) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(1, A);
		exit(-1);
	}
	
	int cw = bw;
	int ch = ah;
	
	float *C = (float *)malloc(cw * ch * sizeof(float));
	if (!C) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(2, A, B);
		exit(-1);
	}
	
	srand(time(NULL));
	int na = ah * aw;
	for (int i = 0; i < na; i++) {
		A[i] = (float)rand() / RAND_MAX;
	}
	
	int nb = bh * bw;
	for (int i = 0; i < nb; i++) {
		B[i] = (float)rand() / RAND_MAX;
	}
	
	int nc = ch * cw;
	for (int i = 0; i < nc; i++) {
		C[i] = 1;
	}
	
	gemm(0, 0, ch, cw, aw, 0.65, A, aw, B, bw, 0.089, C, cw);
	
	FILE *fp = fopen("matrix.txt", "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		mmfree(3, A, B, C);
		exit(-1);
	}
	
	for (int i = 0; i < ah; i++) {
		for (int j = 0; j < aw; j++) {
			fprintf(fp, "%.8f ", A[i * aw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fputs("\n", fp);
	for (int i = 0; i < bh; i++) {
		for (int j = 0; j < bw; j++) {
			fprintf(fp, "%.8f ", B[i * bw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fputs("\n", fp);
	for (int i = 0; i < ch; i++) {
		for (int j = 0; j < cw; j++) {
			fprintf(fp, "%.8f ", C[i * cw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fclose(fp);
	mmfree(3, A, B, C);
}

void test_activate(int argc, char *argv[])
{
	float output[16];
	srand(time(NULL));
	for (int i = 0; i < 16; i++) {
		output[i] = 2 * (rand() / (double)RAND_MAX - 0.5);
		printf("%.5f ", output[i]);
	}
	
	activate(output, 16, RELU);
	
	printf("\n");
	for (int i = 0; i < 16; i++) {
		printf("%.5f ", output[i]);
	}
}

void test_convolutional_layer(int argc, char *argv[])
{
	dim3 input_size = {26, 26, 3};
	float *input = (float *)malloc(input_size.w * input_size.h * input_size.c * sizeof(float));
	if (!input) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	dim3 output_size;
	void *layers[] = {
		make_convolutional_layer(LINEAR, input_size, 3, 512, 1, 1, 1, 0, &output_size)};
	
	convnet *net = convnet_create(layers, 1);
	convnet_architecture(net);
	
	convolutional_layer *layer = (convolutional_layer *)layers[0];
	
	srand(time(NULL));
	for (int i = 0; i < layer->ninputs; ++i) {
		input[i] = (rand() / (double)RAND_MAX - 0.5) * 2;
	}
	
	int size = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->nfilters; ++i) {
		for (int j = 0; j < size; ++j) {
			layer->weights[i * size + j] = 1;
		}
	}
	
	layer->input = input;
	forward_convolutional_layer(layer, net);
	
	FILE *fp = fopen("convolution.txt", "w");
	
	for (int c = 0; c < input_size.c; c++) {
		for (int y = 0; y < input_size.h; y++) {
			for (int x = 0; x < input_size.w; x++) {
				int id = c * input_size.w * input_size.h + y * input_size.w + x;
				if (layer->input[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->input[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fputs("-----------------------------------------\n", fp);
	fputs("-----------------------------------------\n", fp);
	for (int c = 0; c < output_size.c; c++) {
		for (int y = 0; y < output_size.h; y++) {
			for (int x = 0; x < output_size.w; x++) {
				int id = c * output_size.w * output_size.h + y * output_size.w + x;
				if (layer->output[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->output[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fclose(fp);
	convnet_destroy(net);
	free(input);
}

void test_maxpool_layer(int argc, char *argv[])
{
	dim3 input_size = {27, 27, 3};
	float *input = (float *)malloc(input_size.w * input_size.h * input_size.c * sizeof(float));
	if (!input) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	dim3 output_size;
	void *layers[] = {make_maxpool_layer(input_size, 3, 3, 0, 1, &output_size)};
	
	convnet *net = convnet_create(layers, 1);
	convnet_architecture(net);
	
	maxpool_layer *layer = (maxpool_layer *)layers[0];
	
	srand(time(NULL));
	for (int i = 0; i < layer->ninputs; ++i) {
		input[i] = (rand() / (double)RAND_MAX - 0.5) * 2;
	}
	
	layer->input = input;
	forward_maxpool_layer(layer, net);
	
	FILE *fp = fopen("maxpool.txt", "w");
	
	for (int c = 0; c < input_size.c; c++) {
		for (int y = 0; y < input_size.h; y++) {
			for (int x = 0; x < input_size.w; x++) {
				int id = c * input_size.w * input_size.h + y * input_size.w + x;
				if (layer->input[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->input[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fputs("-----------------------------------------\n", fp);
	fputs("-----------------------------------------\n", fp);
	for (int c = 0; c < output_size.c; c++) {
		for (int y = 0; y < output_size.h; y++) {
			for (int x = 0; x < output_size.w; x++) {
				int id = c * output_size.w * output_size.h + y * output_size.w + x;
				if (layer->output[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->output[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fclose(fp);
	convnet_destroy(net);
	free(input);
}

void test_mset(int argc, char *argv[])
{
	float X[128];
	float val = 3.14159;
	
	mset((char *const)X, sizeof(X), (const char *const)&val, sizeof(float));
	
	for (int i = 0; i < 128; ++i) {
		printf("%.5f ", X[i]);
	}
}

void test_mcopy(int argc, char *argv[])
{
	float X[] = {1.111 ,2.222, 3.333, 4.444, 5.555};
	float Y[5];
	
	mcopy((const char *const)X, (char *const)Y, sizeof(X));
	
	for (int i = 0; i < 5; ++i) {
		printf("%f ", Y[i]);
	}
}

void test_bmp(int argc, char *argv[])
{
	BMP *bmp = bmp_read("dog.bmp");
	if (!bmp) {
		fprintf(stderr, "bmp_read[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	printf("bitmap: width %u, height %u, bit_count %u.\n", bmp->width, bmp->height, bmp->bit_count);
	bmp_write(bmp, "girl.bmp");
	bmp_delete(bmp);
}

void test_split(int argc, char *argv[])
{
	BMP *bmp = bmp_read("dog.bmp");
	if (!bmp) {
		fprintf(stderr, "bmp_read[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	printf("bitmap: width %u, height %u, bit_count %u.\n", bmp->width, bmp->height, bmp->bit_count);
	float *splited = calloc(bmp->width * bmp->height * bmp->bit_count, sizeof(float));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		bmp_delete(bmp);
		return;
	}
	
	int nchannels = (bmp->bit_count) >> 3;
	split(bmp->data, splited, bmp->width, bmp->height, nchannels);
	
	FILE *fp = fopen("split.txt", "w");
	for (int i = 0; i < bmp->width * nchannels; ++i) {
		fprintf(fp, "%u ", bmp->data[i]);
	}
	
	fputs("\n\n", fp);
	for (int c = 0; c < nchannels; ++c) {
		for (int i = 0; i < bmp->width; ++i) {
			fprintf(fp, "%.0f ", splited[i + c * bmp->width * bmp->height]);
		}
		fputs("\n", fp);
	}
	
	char *red = calloc(bmp->width * bmp->height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		free(splited);
		bmp_delete(bmp);
		return;
	}
	
	for (int i = 0; i < bmp->width * bmp->height; ++i) {
		red[i] = (char)splited[i];
	}
	
	BMP *red_bmp = bmp_create(red, bmp->width, bmp->height, 8);
	bmp_write(red_bmp, "red.bmp");
	
	fclose(fp);
	free(splited);
	bmp_delete(bmp);
	bmp_delete(red_bmp);
}