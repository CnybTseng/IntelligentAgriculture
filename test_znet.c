#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "znet.h"
#include "im2col.h"
#include "zutils.h"
#include "gemm.h"
#include "activation.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "resample_layer.h"
#include "bitmap.h"
#include "image.h"
#include "list.h"
#include "box.h"
#include "agriculture.names"
#ifdef __ARM_NEON__
	#include <arm_neon.h>
#endif
#ifdef NNPACK
#	include "nnpack.h"
#endif
#include "winograd_convolution.h"
#include "cl_wrapper.h"

typedef struct {
	bitmap *original;
	image *standard;
} test_image;

#ifdef __INTEL_SSE__
extern void split_channel_sse(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
#endif
#ifdef __ARM_NEON__
extern void split_channel_neon(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
extern void resize_image_neon(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                              int dst_w, int dst_h, int nchannels);
extern void embed_image_neon(unsigned char *src, image *dst, int src_w, int src_h);	
extern void make_bilinear_interp_table(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                                       short *y_tab, unsigned short *dx_tab, unsigned short *dy_tab);	
extern void package_neighbor_pixle(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                                   short *y_tab, unsigned char *src, unsigned char *pack);	
extern void resize_image_neon_faster(unsigned char *pack, unsigned char *dst, int dst_w, int dst_h,
                                     int nchannels, unsigned short *dx_tab, unsigned short *dy_tab);								   
void test_resize_faster(int argc, char *argv[]);
#endif
#ifdef OPENCL
cl_wrapper wrapper;
#endif
extern void resize_image_hv(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                            int dst_w, int dst_h, int nchannels);
test_image load_test_image(int argc, char *argv[], int std_width, int std_height);
void draw_detections(bitmap *bmp, list *detections, char *names[], float thresh);
void test_multi_free(int argc, char *argv[]);
void test_yolov3_tiny(int argc, char *argv[]);
void test_im2col(int argc, char *argv[]);
void test_gemm(int argc, char *argv[]);
void test_activate(int argc, char *argv[]);
void test_convolutional_layer(int argc, char *argv[]);
void test_maxpool_layer(int argc, char *argv[]);
void test_mset(int argc, char *argv[]);
void test_mcopy(int argc, char *argv[]);
void test_bmp(int argc, char *argv[]);
void test_split(int argc, char *argv[]);
void test_resize(int argc, char *argv[]);
void test_embed(int argc, char *argv[]);
void test_standard(int argc, char *argv[]);
void test_list(int argc, char *argv[]);
void test_split_sse(int argc, char *argv[]);
void test_split_compare(int argc, char *argv[]);
void test_resize_compare(int argc, char *argv[]);
void test_activate_neon(int argc, char *argv[]);
void test_nnpack(int argc, char *argv[]);
void test_winograd_weight_transformation(int argc, char *argv[]);
void test_winograd_input_transformation(int argc, char *argv[]);
void test_winograd_convolution(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	test_winograd_convolution(argc, argv);
	
	return 0;
}

test_image load_test_image(int argc, char *argv[], int std_width, int std_height)
{
	test_image ti = {NULL, NULL};
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return ti;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return ti;
	}
	
	int rsz_width, rsz_height;
	if (std_width / (float)width < std_height / (float)height) {
		rsz_width = std_width;
		rsz_height = (int)(height * std_width / (float)width);
	} else {
		rsz_width = (int)(width * std_height / (float)height);
		rsz_height = std_height;
	}
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		free(splited);
		return ti;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	resize_image(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	
	image *standard = create_image(std_width, std_height, nchannels);
	if (!standard) {
		fprintf(stderr, "create_image[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		free(splited);
		free(rsz_splited);
		return ti;
	}
	
	set_image(standard, 0.5);
	embed_image(rsz_splited, standard, rsz_width, rsz_height);
	
	free(splited);
	free(rsz_splited);
	
	ti.original = bmp;
	ti.standard = standard;
	
	return ti;
}

void draw_detections(bitmap *bmp, list *detections, char *names[], float thresh)
{
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	int nchannels = bit_count >> 3;
	node *n = detections->head;
	int color[3] = {0, 255, 255};
	while (n) {
		detection *det = (detection *)n->val;
		if (det->objectness < thresh) {
			n = n->next;
			continue;
		}
		
		printf("objectness:%.5f ", det->objectness);
		int maybe = 0;
		for (int i = 0; i < det->classes; ++i) {
			if (det->probabilities[i] < thresh) continue;
			if (maybe > 0) printf(",");
			printf("%s:%.0f%%", names[i], det->probabilities[i] * 100);
			++maybe;
		}
		
		if (maybe) printf("\n");
		int left    = (int)((det->bbox.x - det->bbox.w / 2) * width);		
		int right   = (int)((det->bbox.x + det->bbox.w / 2) * width);
		int _top    = (int)((det->bbox.y - det->bbox.h / 2) * height);
		int _bottom = (int)((det->bbox.y + det->bbox.h / 2) * height);
		int top = height - 1 - _bottom;
		int bottom = height - 1 - _top;
		
		if (left < 0) left = 0;
		if (left > width - 1) left = width - 1;
		if (right < 0) right = 0;
		if (right > width - 1) right = width - 1;
		if (top < 0) top = 0;
		if (top > height - 1) top = height - 1;
		if (bottom < 0) bottom = 0;
		if (bottom > height - 1) bottom = height - 1;
		
		for (int c = 0; c < nchannels; ++c) {
			for (int y = top; y < bottom; ++y) {
				data[y * pitch + left * nchannels + c] = color[c];
				data[y * pitch + right * nchannels + c] = color[c];
			}
			
			for (int x = left; x < right; ++x) {
				data[top * pitch + x * nchannels + c] = color[c];
				data[bottom * pitch + x * nchannels + c] = color[c];
			}
		}
		
		n = n->next;
	}
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

void test_yolov3_tiny(int argc, char *argv[])
{	
	if (argc < 2) {
		fprintf(stderr, "Usage: detector [bitmap filename] [thresh]\n");
		return;
	}

#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
#endif	
	
	void *layers[24];
	int nlayers = sizeof(layers) / sizeof(layers[0]);
	dim3 output_size;
	
	int bigger_mask[] = {3, 4, 5};
	int smaller_mask[] = {0, 1, 2};
	int anchor_boxes[] = {61,117, 62,191, 199,118, 128,195, 92,293, 191,291};
	const int scales = 3;
	const int classes = 1;
	const int object_tensor_depth = (4 + 1 + classes) * scales;
	
	dim3 input_size = {416, 416, 3};
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
	
	znet *net = znet_create(layers, nlayers, "agriculture.weights");
	if (!net) return;
	
	znet_architecture(net);
	
	test_image ti = load_test_image(argc, argv, 416, 416);
	if (!ti.original || !ti.standard) {
		znet_destroy(net);
		return;
	}
		
	bitmap *original = ti.original;
	image *standard = ti.standard;
	
	unsigned char *red = calloc(standard->w * standard->h, sizeof(unsigned char));
	for (int i = 0; i < standard->w * standard->h; ++i)
		red[i] = (unsigned char)(standard->data[i] * 255);
	bitmap *red_bmp = create_bmp((const char *)red, standard->w, standard->h, 8);
	save_bmp(red_bmp, "standard.bmp");
	delete_bmp(red_bmp);
	free(red);
	
	int N = 1;
	if (argc > 3) N = atoi(argv[3]);
	printf("inference iterations %d\n", N);

#ifdef NNPACK	
	znet_inference(net, standard);
#endif	
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		printf("------------------------------------------------------\n");
		znet_inference(net, standard);
	}
	gettimeofday(&t2, NULL);
	printf("time: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	float thresh = 0.5f;
	if (argc > 2) thresh = atof(argv[2]);
	
	int width = get_bmp_width(original);
	int height = get_bmp_height(original);
	
	list *detections = get_detections(net, thresh, width, height);	
	list *bests = soft_nms(detections, 2.5);
	
	draw_detections(original, bests, names, thresh);
	save_bmp(original, "detections.bmp");
	
	free_detections(bests);
	free_detections(detections);
	delete_bmp(original);
	free_image(standard);
	znet_destroy(net);
#ifdef OPENCL
	cl_destroy_wrapper(wrapper);
#endif
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
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
#endif	
	
	int ah = 1024;
	int aw = 1024;
	int bh = 1024;
	int bw = 1024;
	
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
	
	int transa = 0;
	int transb = 0;
	
	int cw, ch;
	if (!transa && !transb) {
		cw = bw;
		ch = ah;
	} else if (!transa && transb) {
		cw = bh;
		ch = ah;
	} else if (transa && !transb) {
		cw = bw;
		ch = aw;
	} else {
		cw = bh;
		ch = aw;
	}
	
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
	
	int N = 10000;
	if (argc > 1) N = atoi(argv[1]);
	printf("gemm iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		gemm(transa, transb, ch, cw, aw, 0.56, A, aw, B, bw, 0.84, C, cw);
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
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
	
#ifdef OPENCL
	cl_destroy_wrapper(wrapper);
#endif
}

void test_activate(int argc, char *argv[])
{
	float output[16];
	srand(time(NULL));
	for (int i = 0; i < 16; i++) {
		output[i] = 2 * (rand() / (double)RAND_MAX - 0.5);
		printf("%.5f ", output[i]);
	}
	
	activate(output, 16, LEAKY);
	
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
	
	znet *net = znet_create(layers, 1, "coco.weights");
	znet_architecture(net);
	
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
	znet_destroy(net);
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
	
	znet *net = znet_create(layers, 1, "coco.weights");
	znet_architecture(net);
	
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
	znet_destroy(net);
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
	bitmap *bmp = read_bmp("dog.bmp");
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	
	save_bmp(bmp, "girl.bmp");
	delete_bmp(bmp);
}

void test_split(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	
	FILE *fp = fopen("split_channel.txt", "w");
	for (int i = 0; i < width * nchannels; ++i) {
		fprintf(fp, "%u ", data[i + width * (height - 1) * 3]);
	}
	
	fputs("\n\n", fp);
	for (int c = 0; c < nchannels; ++c) {
		for (int i = 0; i < width; ++i) {
			fprintf(fp, "%u ", splited[i + c * width * height]);
		}
		fputs("\n", fp);
	}
	
	char *red = calloc(width * height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < width * height; ++i) {
		red[i] = splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, width, height, 8);
	save_bmp(red_bmp, "splited.bmp");
	
	fclose(fp);
	free(splited);
	free(red);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_resize(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);	
	split_channel(data, splited, pitch, width, height);
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		resize_image_hv(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
	char *red = calloc(rsz_width * rsz_height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < rsz_width * rsz_height; ++i) {
		red[i] = rsz_splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, rsz_width, rsz_height, 8);
	save_bmp(red_bmp, "resized.bmp");
	
	free(red);
	free(splited);
	free(rsz_splited);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_embed(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	resize_image(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	
	image *standard = create_image(416, 416, nchannels);
	if (!standard) {
		fprintf(stderr, "create_image[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("embed iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		embed_image(rsz_splited, standard, rsz_width, rsz_height);
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	memset(standard->data, 0, standard->w * standard->h * nchannels * sizeof(float));

	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
#ifdef __ARM_NEON__
		embed_image_neon(rsz_splited, standard, rsz_width, rsz_height);
#endif
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	char *red = calloc(standard->w * standard->h, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		free_image(standard);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < standard->w * standard->h; ++i) {
		red[i] = (char)(standard->data[i] * 255);
	}
	
	bitmap *red_bmp = create_bmp(red, standard->w, standard->h, 8);
	save_bmp(red_bmp, "standard.bmp");
	
	free(red);
	free(splited);
	free(rsz_splited);
	free_image(standard);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_standard(int argc, char *argv[])
{
	image *standard = create_image(416, 416, 3);
	FILE *fp = fopen("standard.bin", "rb");
	fread(standard->data, sizeof(float), 416 * 416 * 3, fp);
	fclose(fp);
	
	char *red = calloc(standard->w * standard->h, sizeof(char));	
	for (int i = 0; i < standard->w * standard->h; ++i) {
		red[i] = (char)(standard->data[i] * 255);
	}
	
	bitmap *bmp = create_bmp(red, standard->w, standard->h, 8);
	save_bmp(bmp, "red.bmp");
	
	free(red);
	delete_bmp(bmp);
	free_image(standard);
}

void test_list(int argc, char *argv[])
{
	list *detections = make_list();
	if (!detections) return;
	
	int bx = 5;
	int by = 6;
	int bw = 7;
	int bh = 8;
	
	srand(time(NULL));
	for (int i = 0; i < 10; ++i) {
		detection *det = list_alloc_mem(sizeof(detection));
		if (!det) break;
		det->bbox.x = (i + 1) * bx;
		det->bbox.y = (i + 1) * by;
		det->bbox.w = (i + 1) * bw;
		det->bbox.h = (i + 1) * bh;
		det->classes = 80;
		det->probabilities = calloc(det->classes, sizeof(float));
		if (!det->probabilities) break;
		for (int j = 0; j < det->classes; ++j)
			det->probabilities[j] = rand() / (double)RAND_MAX;
		det->objectness = rand() / (double)RAND_MAX;
		if (list_add_tail(detections, det)) break;
	}
		
	int count = 0;
	node *nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		printf("%d   %.2f:%.2f:%.2f:%.2f   %d   ", ++count, det->bbox.x, det->bbox.y, det->bbox.w,
			det->bbox.h, det->classes);
		for (int i = 0; i < det->classes; ++i)
			printf("%.2f:", det->probabilities[i]);
		printf("   %f  %p\n\n\n", det->objectness, nd);
		nd = nd->next;
	}
	
	printf("head=%p, tail=%p\n", detections->head, detections->tail);
	printf("//////////////////////////////////////////////////\n");
	detection deleted[3];
	for (int i = 0; i < 3; ++i) {
		deleted[i].bbox.x = (i + 3) * bx;
		deleted[i].bbox.y = (i + 3) * by;
		deleted[i].bbox.w = (i + 3) * bw;
		deleted[i].bbox.h = (i + 3) * bh;
		list_del_node(detections, &deleted[i], equ_val, free_val);
	}
	
	count = 0;
	nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		printf("%d   %.2f:%.2f:%.2f:%.2f   %d   ", ++count, det->bbox.x, det->bbox.y, det->bbox.w,
			det->bbox.h, det->classes);
		for (int i = 0; i < det->classes; ++i)
			printf("%.2f:", det->probabilities[i]);
		printf("   %f  %p\n\n\n", det->objectness, nd);
		nd = nd->next;
	}
	
	printf("head=%p, tail=%p\n", detections->head, detections->tail);
	nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		if (det->probabilities) {
			free(det->probabilities);
			det->probabilities = NULL;
		}
		nd = nd->next;
	}
	
	list_clear(detections);
}

void test_split_sse(int argc, char *argv[])
{
#ifdef __INTEL_SSE__
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	char *splited = calloc(width * height * 3, 1);
	
	split_channel_sse(data, (unsigned char *)splited, pitch, width, height);

	bitmap *red = create_bmp(splited + 0 * width * height, width, height, 8);
	save_bmp(red, "splited.bmp");
	
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
#endif
}

void test_split_compare(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	unsigned char *splited = calloc(width * height * 3, 1);
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("split iterations %d\n", N);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		split_channel(data, splited, pitch, width, height);
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
#ifdef __INTEL_SSE__
		split_channel_sse(data, splited, pitch, width, height);
#endif
#ifdef __ARM_NEON__
		split_channel_neon(data, splited, pitch, width, height);
#endif
	}
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	bitmap *red = create_bmp((char *)splited, width, height, 8);
	save_bmp(red, "splited.bmp");
	
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
}

void test_resize_compare(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	unsigned char *data = get_bmp_data(bmp);
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * 3, sizeof(unsigned char));
	split_channel(data, splited, pitch, width, height);
	bitmap *red = create_bmp((char *)splited, width, height, 8);
	save_bmp(red, "splited.bmp");

	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *resized = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("resize iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		resize_image(splited, resized, width, height, rsz_width, rsz_height, nchannels);
	}
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	memset(resized, 0, rsz_width * rsz_height * nchannels * sizeof(unsigned char));
	
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
#ifdef __ARM_NEON__
		resize_image_neon((uint8_t *)splited, resized, width, height, rsz_width, rsz_height, nchannels);
#endif
	}
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	bitmap *resized_bmp = create_bmp((char *)resized, rsz_width, rsz_height, 8);
	save_bmp(resized_bmp, "resized.bmp");
	
	free(resized);
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
	delete_bmp(resized_bmp);
}

#ifdef __ARM_NEON__
void test_resize_faster(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);	
	split_channel(data, splited, pitch, width, height);
	
	short *x_tab = calloc(rsz_width * rsz_height, sizeof(short));
	short *y_tab = calloc(rsz_width * rsz_height, sizeof(short));
	unsigned short *dx_tab = calloc(rsz_width * rsz_height, sizeof(unsigned short));
	unsigned short *dy_tab = calloc(rsz_width * rsz_height, sizeof(unsigned short));
	unsigned char *pack = calloc(rsz_width * rsz_height * 4 * nchannels, sizeof(unsigned char));
	
	make_bilinear_interp_table(width, height, rsz_width, rsz_height, x_tab, y_tab, dx_tab, dy_tab);
		
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("faster resize iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		package_neighbor_pixle(width, height, rsz_width, rsz_height, x_tab, y_tab, splited, pack);
		resize_image_neon_faster(pack, rsz_splited, rsz_width, rsz_height, nchannels, dx_tab, dy_tab);
	}
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
	char *red = calloc(rsz_width * rsz_height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < rsz_width * rsz_height; ++i) {
		red[i] = rsz_splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, rsz_width, rsz_height, 8);
	save_bmp(red_bmp, "resized.bmp");
	
	free(x_tab);
	free(y_tab);
	free(dx_tab);
	free(dy_tab);
	free(pack);
	free(red);
	free(splited);
	free(rsz_splited);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}
#endif

void test_activate_neon(int argc, char *argv[])
{
	float X[16];
	srand(time(NULL));
	for (int i = 0; i < 16; ++i) {
		X[i] = 2 * (rand() / (float)RAND_MAX - 0.5) + 0.5;
		printf("%.5f ", X[i]);
	}
	printf("\n");
	
	activate(X, 16, LEAKY);
	for (int i = 0; i < 16; ++i)
		printf("%.5f ", X[i]);
	printf("\n");
}

#ifdef NNPACK
void test_nnpack(int argc, char *argv[])
{
	pthreadpool_t threadpool = pthreadpool_create(1);
	if (nnp_initialize() != nnp_status_success) {
		fprintf(stderr, "nnp_initialize fail!\n");
		pthreadpool_destroy(threadpool);
		return;
	}
	
	int xreso = 8;
	int yreso = 8;
	struct nnp_size input_size = {xreso, yreso};
	struct nnp_padding input_padding = {1, 1, 1, 1};
	struct nnp_size kernel_size = {3, 3};
	struct nnp_size stride = {1, 1};

	srand(time(NULL));
	
	float *input = calloc(xreso * yreso * 3, sizeof(float));
	for (int c = 0; c < 3; ++c) {
		float *ptr = input + c * xreso * yreso;
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				ptr[y * xreso + x] = 2 * (rand() / (float)RAND_MAX - 0.5) + 0.5;
			}
		}
	}
	
	int nfilters = 64;
	float *weights = calloc(3 * 3 * 3 * nfilters, sizeof(float));
	for (int i = 0; i < 3 * 3 * 3 * nfilters; ++i) {
		weights[i] = 1;
	}
	
	float *bias = calloc(nfilters, sizeof(float));
	float *output = calloc(xreso * yreso * nfilters, sizeof(float));

	nnp_convolution_inference(
		nnp_convolution_algorithm_implicit_gemm,
		nnp_convolution_transform_strategy_tuple_based,
		3,
		nfilters,
		input_size,
		input_padding,
		kernel_size,
		stride,
		input,
		weights,
		bias,
		output,
		NULL,
		NULL,
		nnp_activation_identity,
		NULL,
		threadpool,
		NULL
	);

	FILE *fp = fopen("conv.txt", "w");
	for (int c = 0; c < 3; ++c) {
		float *ptr = input + c * xreso * yreso;
		fprintf(fp, "channel = %d\n", c);
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				fprintf(fp, "%.5f ", ptr[y * xreso + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n\n", fp);
	}
	
	fputs("===============================================\n", fp);
	for (int c = 0; c < nfilters; ++c) {
		float *ptr = output + c * xreso * yreso;
		fprintf(fp, "channel = %d\n", c);
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				fprintf(fp, "%.5f ", ptr[y * xreso + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n\n", fp);
	}
	
	fclose(fp);
	free(input);
	free(weights);
	free(bias);
	free(output);
	pthreadpool_destroy(threadpool);
	nnp_deinitialize();
}
#endif

void test_winograd_weight_transformation(int argc, char *argv[])
{
#ifdef OPENCL
	int save_result = 0;
	if (argc > 2) {
		save_result = atoi(argv[2]);
	}
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
	
	const int filter_size = 3;
	const int filter_channels = 512;
	const int nfilters = 1024;
	float *weights = calloc(filter_size * filter_size * filter_channels * nfilters, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < filter_size * filter_size * filter_channels * nfilters; ++i) {
		weights[i] = rand() / (double)RAND_MAX;
	}
	
	float *biases = calloc(nfilters, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < nfilters; ++i) {
		biases[i] = rand() / (double)RAND_MAX;
	}
	
	weight_transform_context *context = create_weight_transform_context(F4x4_3x3, filter_channels, nfilters);
	
	int transformed_weight_image_width, transformed_weight_image_height;
	float *transformed_weights = NULL;
	if (save_result) {
		get_transformed_weight_image_size(context, &transformed_weight_image_width, &transformed_weight_image_height);
		transformed_weights = calloc((transformed_weight_image_width << 2) * transformed_weight_image_height, sizeof(float));
	}
	
	int N = 1;
	if (argc > 1) N = atoi(argv[1]);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		transform_weight(context, weights, biases, transformed_weights);
	}
	gettimeofday(&t2, NULL);
	printf("transform_weight: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	free_weight_transform_context(context);
	
	if (save_result) {
		save_volume(weights, 3, 3, filter_channels * nfilters, "weights.txt");
		save_volume(transformed_weights, transformed_weight_image_width << 2, transformed_weight_image_height, 1, "transformed_weights.txt");
		free(transformed_weights);
	}
	
	free(weights);
	free(biases);
	
	cl_destroy_wrapper(wrapper);
#endif
}

void test_winograd_input_transformation(int argc, char *argv[])
{
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);

	const int input_width = 13;
	const int input_height = 13;
	const int input_channels = 512;
	const int stride = 1;
	const int padding = 1;
	
	float *input = calloc(input_width * input_height * input_channels, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < input_width * input_height * input_channels; ++i) {
		input[i] = rand() / (double)RAND_MAX;
	}
	
	save_volume(input, input_width, input_height, input_channels, "input.txt");
	
	input_transform_context *context = create_input_transform_context(F4x4_3x3, input_width, input_height,
		input_channels, stride, padding);
	
	int transformed_input_width, transformed_input_height;
	get_transformed_input_image_size(context, &transformed_input_width, &transformed_input_height);
	float *transformed_input = calloc((transformed_input_width << 2) * transformed_input_height, sizeof(float));
	
	int N = 1;
	if (argc > 1) N = atoi(argv[1]);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		transform_input(context, input, transformed_input);
	}
	gettimeofday(&t2, NULL);
	printf("transform_input: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	save_volume(transformed_input, transformed_input_width << 2, transformed_input_height, 1, "transformed_input.txt");
	
	free_input_transform_context(context);
	free(input);
	free(transformed_input);

	cl_destroy_wrapper(wrapper);
#endif	
}

void test_winograd_convolution(int argc, char *argv[])
{
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
	cl_print_device_info(wrapper, CL_DEVICE_EXTENSIONS);

	int filter_channels_test_group[] = {  3,  16,  32,  64, 128, 256,  512, 256, 384};
	int nfilters_test_group[] =        { 16,  32,  64, 128, 256, 512, 1024, 512, 256};
	int input_size_test_group[] =      {416, 208, 104,  52,  26,  13,   13,  13,  26};
	
	int save_result = 0;
	if (argc > 1)
		save_result = atoi(argv[1]);
	
	float total_duration[3] = {0, 0, 0};
	for (int g = 0;  g < 9; ++g) {
		float *transformed_weights = NULL;
		float *transformed_input = NULL;
		float *output = NULL;
		float *inverse_transformed_output = NULL;
		
		const int filter_channels = filter_channels_test_group[g];
		const int nfilters = nfilters_test_group[g];
		
		float *weights = calloc(3 * 3 * filter_channels * nfilters, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < 3 * 3 * filter_channels * nfilters; ++i)
			weights[i] = rand() / (double)RAND_MAX;
		
		float *biases = calloc(nfilters, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < nfilters; ++i)
			biases[i] = rand() / (double)RAND_MAX;
		
		const int input_width = input_size_test_group[g];
		const int input_height = input_size_test_group[g];
		const int input_channels = filter_channels;
		const int stride = 1;
		const int padding = 1;
		
		float *input = calloc(input_width * input_height * input_channels, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < input_width * input_height * input_channels; ++i)
			input[i] = 0.1f * (rand() / (double)RAND_MAX - 0.5f);
				
		printf("\ntest configuration: input size %d, filter channels %d, number of filters %d\n", input_size_test_group[g],
			filter_channels_test_group[g], nfilters_test_group[g]);

		weight_transform_context *wtc = create_weight_transform_context(F4x4_3x3, filter_channels, nfilters);
		
		int transformed_weight_image_width, transformed_weight_image_height;
		if (save_result) {
			get_transformed_weight_image_size(wtc, &transformed_weight_image_width, &transformed_weight_image_height);
			transformed_weights = calloc((transformed_weight_image_width << 2) * transformed_weight_image_height, sizeof(float));
		}
		transform_weight(wtc, weights, biases, transformed_weights);
		if (save_result) {
			save_volume(weights, 3, 3, filter_channels * nfilters, "weights.txt");
			save_volume(biases, nfilters, 1, 1, "biases.txt");
			save_volume(transformed_weights, transformed_weight_image_width << 2, transformed_weight_image_height, 1, "transformed_weights.txt");
			free(transformed_weights);
		}

		input_transform_context *itc = create_input_transform_context(F4x4_3x3, input_width, input_height,
			input_channels, stride, padding);
		
		int transformed_input_width, transformed_input_height;		
		if (save_result) {
			get_transformed_input_image_size(itc, &transformed_input_width, &transformed_input_height);
			transformed_input = calloc((transformed_input_width << 2) * transformed_input_height, sizeof(float));
		}
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
		transform_input(itc, input, transformed_input);
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[0] += duration;
		printf("GPU & CPU, transform_input: %f ms, total %fms.\n", duration, total_duration[0]);
		if (save_result) {
			save_volume(input, input_width, input_height, input_channels, "inputs.txt");
			save_volume(transformed_input, transformed_input_width << 2, transformed_input_height, 1, "transformed_input.txt");
			free(transformed_input);
		}
		
		matrix_multiplication_context *mmc = create_matrix_multiplication_context(wtc, itc);
		
		int output_image_width, output_image_height;
		if (save_result) {
			get_transformed_output_image_size(mmc, &output_image_width, &output_image_height);
			output = calloc((output_image_width << 2) * output_image_height, sizeof(float));
		}
		gettimeofday(&t1, NULL);
		multiply_transformed_matrix(mmc, output);
		gettimeofday(&t2, NULL);
		duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[1] += duration;
		printf("GPU & CPU, multiply_transformed_matrix: %f ms, total %fms.\n", duration, total_duration[1]);
		if (save_result) {
			save_volume(output, output_image_width << 2, output_image_height, 1, "output.txt");
			free(output);
		}
	
		output_inverse_transform_context *oitc = create_output_inverse_transform_context(mmc);
		
		int inverse_transformed_output_image_width, inverse_transformed_output_image_height;
		if (save_result) {
			get_inverse_transformed_output_image_size(oitc, &inverse_transformed_output_image_width,
				&inverse_transformed_output_image_height);
			inverse_transformed_output = calloc((inverse_transformed_output_image_width << 2) *
				inverse_transformed_output_image_height, sizeof(float));
		}
		gettimeofday(&t1, NULL);
		inverse_transform_output(oitc, inverse_transformed_output);
		gettimeofday(&t2, NULL);
		duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[2] += duration;
		printf("GPU & CPU, inverse_transform_output: %f ms, total %fms.\n", duration, total_duration[2]);
		if (save_result) {
			save_volume(inverse_transformed_output, inverse_transformed_output_image_width << 2, inverse_transformed_output_image_height,
				1, "inverse_transformed_output.txt");
			free(inverse_transformed_output);
		}
		
		free_input_transform_context(itc);
		free_weight_transform_context(wtc);
		free_matrix_multiplication_context(mmc);
		free_output_inverse_transform_context(oitc);
		
		if (save_result) {
			dim3 input_size = {input_width, input_height, filter_channels};
			dim3 output_size;
			void *layers[] = {make_convolutional_layer(LEAKY, input_size, 3, nfilters, 1, 1, 1, 0, &output_size)};
			znet *net = znet_create(layers, 1, "coco.weights");
			convolutional_layer *layer = (convolutional_layer *)layers[0];
			memcpy(layer->weights, weights, 3 * 3 * filter_channels * nfilters * sizeof(float));
			for (int i = 0; i < nfilters; ++i) layer->biases[i] = biases[i];
			layer->input = input;
			forward_convolutional_layer(layer, net);
			save_volume(layer->output, output_size.w, output_size.h, output_size.c, "gemm_conv.txt");
			znet_destroy(net);
		}
		
		free(weights);
		free(biases);
		free(input);
	}

	cl_destroy_wrapper(wrapper);
#endif	
}