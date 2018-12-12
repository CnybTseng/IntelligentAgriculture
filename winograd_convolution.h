#ifndef _WINOGRAD_CONVOLUTION_H_
#define _WINOGRAD_CONVOLUTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	F6x6_3x3
} WINOGRAD_CONV_TYPE;

int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv);
void transform_weight(WINOGRAD_CONV_TYPE conv, float *weights, int filter_size, int filter_channels,
                      int nfilters, float *transformed_weights);
void winograd_convolution();

#ifdef __cplusplus
}
#endif

#endif