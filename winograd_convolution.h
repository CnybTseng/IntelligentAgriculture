#ifndef _WINOGRAD_CONVOLUTION_H_
#define _WINOGRAD_CONVOLUTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	F6x6_3x3, F4x4_3x3, F2x2_3x3
} WINOGRAD_CONV_TYPE;

#ifdef OPENCL
struct weight_transform_context;
typedef struct weight_transform_context weight_transform_context;

struct input_transform_context;
typedef struct input_transform_context input_transform_context;

struct matrix_multiplication_context;
typedef struct matrix_multiplication_context matrix_multiplication_context;

struct output_inverse_transform_context;
typedef struct output_inverse_transform_context output_inverse_transform_context;
#endif

int get_image_tile_size(WINOGRAD_CONV_TYPE conv);
int get_tile_output_size(WINOGRAD_CONV_TYPE conv);
#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv, int filter_channels,
	int nfilters);
void transform_weight(weight_transform_context *context, float *weights, float *transformed_weights);
void free_weight_transform_context(weight_transform_context *context);
input_transform_context *create_input_transform_context(WINOGRAD_CONV_TYPE conv, int input_width,
	int input_height, int input_channels, int stride, int padding);
void get_transformed_input_size(input_transform_context *context, int *width, int *height, int *array_size);
void transform_input(input_transform_context *context, float *input, float *transformed_input);
void free_input_transform_context(input_transform_context *context);
matrix_multiplication_context *create_matrix_multiplication_context(WINOGRAD_CONV_TYPE conv,
	weight_transform_context *wtc, input_transform_context *itc);
void muliply_transformed_matrix(matrix_multiplication_context *context, float *output);
void free_matrix_multiplication_context(matrix_multiplication_context *context);
output_inverse_transform_context *create_output_inverse_transform_context(WINOGRAD_CONV_TYPE conv,
	matrix_multiplication_context *mmc);
void inverse_transform_output(output_inverse_transform_context *context, float *inverse_transformed_output);
void free_output_inverse_transform_context(output_inverse_transform_context *context);
#endif

#ifdef __cplusplus
}
#endif

#endif