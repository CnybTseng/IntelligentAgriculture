#include <stdio.h>
#include "winograd_convolution.h"
#include "gemm.h"

static float weight_transformation_matrix_f6x6_3x3[] = {
	     1,       0,       0,
	-2/9.0,  -2/9.0,  -2/9.0,
	-2/9.0,   2/9.0,  -2/9.0,
	1/90.0,  1/45.0,  2/45.0,
	1/90.0, -1/45.0,  2/45.0,
	1/45.0,  1/90.0, 1/180.0,
	1/45.0, -1/90.0, 1/180.0,
	     0,       0,       1
};

static float weight_transformation_matrix_transposed_f6x6_3x3[] = {
	1, -2/9.0, -2/9.0, 1/90.0,  1/90.0, 1/45.0,   1/45.0, 0,
	0, -2/9.0,  2/9.0, 1/45.0, -1/45.0, 1/90.0,  -1/90.0, 0,
	0, -2/9.0, -2/9.0, 2/45.0,  2/45.0, 1/180.0, 1/180.0, 1
};
#if 0
static float input_transformation_matrix_f6x6_3x3[] = {
	1,      0, -21/4.0,       0,  21/4.0,       0, -1, 0,
	0,      1,       1, -17/4.0, -17/4.0,       1,  1, 0,
	0,     -1,       1,  17/4.0, -17/4.0,      -1,  1, 0,
	0,  1/2.0,   1/4.0,  -5/2.0,  -5/4.0,       2,  1, 0,
	0, -1/2.0,   1/4.0,   5/2.0,  -5/4.0,      -2,  1, 0,
	0,      2,       4,  -5/2.0,      -5,   1/2.0,  1, 0,
	0,     -2,       4,   5/2.0,      -5,  -1/2.0,  1, 0,
	0,     -1,       0,  21/4.0,       0, -21/4.0,  0, 1
};

static float input_transformation_matrix_transposed_f6x6_3x3[] = {
	      1,       0,       0,      0,      0,      0,      0,       0,
	      0,       1,      -1,  1/2.0, -1/2.0,      2,     -2,      -1,
	-21/4.0,       1,       1,  1/4.0,  1/4.0,      4,      4,       0,
	      0, -17/4.0,  17/4.0, -5/2.0,  5/2.0, -5/2.0,  5/2.0,  21/4.0,
	 21/4.0, -17/4.0, -17/4.0, -5/4.0, -5/4.0,     -5,     -5,       0,
	      0,       1,      -1,      2,     -2,  1/2.0, -1/2.0, -21/4.0,
	     -1,       1,       1,      1,      1,      1,      1,       0,
	      0,       0,       0,      0,      0,      0,      0,       1
};

static float output_inverse_transformation_matrix_f6x6_3x3[] = {
	1, 1,  1,  1,   1, 32,  32, 0,
	0, 1, -1,  2,  -2, 16, -16, 0,
	0, 1,  1,  4,   4,  8,   8, 0,
	0, 1, -1,  8,  -8,  4,  -4, 0,
	0, 1,  1, 16,  16,  2,   2, 0,
	0, 1, -1, 32, -32,  1,  -1, 1
};

static float output_inverse_transformation_matrix_transposed_f6x6_3x3[] = {
	1,    0, 0,  0,  0,   0,
	1,    1, 1,  1,  1,   1,
	1,   -1, 1, -1,  1,  -1,
	1,    2, 4,  8, 16,  32,
	1,   -2, 4, -8, 16, -32,
	32,  16, 8,  4,  2,   1,
	32, -16, 8, -4,  2,  -1,
	0,    0, 0,  0,  0,   1
};
#endif
int get_transformed_weight_matrix_size(WINOGRAD_CONV_TYPE conv)
{
	int filter_size;
	int conv_tile_output_size;
	if (conv == F6x6_3x3) {
		filter_size = 3;
		conv_tile_output_size = 6;
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	return filter_size + conv_tile_output_size - 1;
}

void transform_weight(WINOGRAD_CONV_TYPE conv, float *weights, int filter_size, int filter_channels,
                      int nfilters, float *transformed_weights)
{
	float *G = weight_transformation_matrix_f6x6_3x3;
	float *G_t = weight_transformation_matrix_transposed_f6x6_3x3;
	float *g = weights;
	float *GgG_t = transformed_weights;
	if (conv == F6x6_3x3) {
		const int total = filter_channels * nfilters;
		const int tran_size = get_transformed_weight_matrix_size(conv);
		float Gg[tran_size * filter_size];
		for (int j = 0; j < total; ++j) {
			for (int i = 0; i < tran_size * filter_size; ++i) Gg[i] = 0;
			for (int i = 0; i < tran_size * tran_size; ++i) GgG_t[i] = 0;
			gemm(0, 0, 8, 3, 3, 1, G, 3, g, 3, 0, Gg, 3);
			gemm(0, 0, 8, 8, 3, 1, Gg, 3, G_t, 3, 0, GgG_t, 8);
			g += filter_size * filter_size;
			GgG_t += tran_size * tran_size;
		}
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}