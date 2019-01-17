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
#endif

int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv);
#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv);
void transform_weight(weight_transform_context *context, float *weights, int filter_size,
                      int filter_channels, int nfilters, float *transformed_weights);
void free_weight_transform_context(weight_transform_context *context);
#endif

#ifdef __cplusplus
}
#endif

#endif