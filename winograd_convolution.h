#ifndef _WINOGRAD_CONVOLUTION_H_
#define _WINOGRAD_CONVOLUTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	F6x6_3x3
} WINOGRAD_CONV_TYPE;

#ifdef OPENCL
struct weight_transform_context;
typedef struct weight_transform_context weight_transform_context;

struct input_transform_context;
typedef struct input_transform_context input_transform_context;

struct winograd_convolution_context;
typedef struct winograd_convolution_context winograd_convolution_context;
#endif

int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv);
#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv);
void transform_weight(weight_transform_context *context, float *weights, int filter_size,
                      int filter_channels, int nfilters, float *transformed_weights);
void free_weight_transform_context(weight_transform_context *context);

input_transform_context *create_input_transform_context(int input_width, int input_height,
	int input_channels, int stride, int padding);
void transform_input(input_transform_context *context, float *input);
void free_input_transform_context(input_transform_context *context);

winograd_convolution_context *create_winograd_convolution_context(WINOGRAD_CONV_TYPE conv,
	int input_width, int input_height, int input_channels, int stride, int padding, int output_channels);
void winograd_convolution(weight_transform_context *wt_context, winograd_convolution_context *wc_context,
	float *input, int input_width, int input_height, int input_channels, int stride, int padding,
	int output_channels, float *output);
void free_winograd_convolution_context(winograd_convolution_context *context);
#endif

#ifdef __cplusplus
}
#endif

#endif