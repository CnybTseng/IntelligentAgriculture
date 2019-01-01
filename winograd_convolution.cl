/** @brief Change data storing order from NCHW to CHWN.
 ** @param nchw input data stored in NCHW order.
 ** @param chwn output data stored in CHWN order.
 ** @param n batch size.
 ** @param c number of channels.
 ** @param h height of input tensor.
 ** @param w width of input tensor.
 **/
__kernel
void nchw2chwn(__global float *nchw,
	__global float *chwn, int n, int c, int h, int w)
{
	int global_id = get_global_id(0);
	int batch_id = global_id / (c * h * w);
	if (batch_id < n) {
		int chw = global_id - batch_id * c * h * w;
		chwn[chw * n + batch_id] = nchw[global_id];
	}
}

/** @brief Change data storing order from CHWN to NCHW.
 ** @param chwn input data stored in CHWN order.
 ** @param nchw output data stored in NCHW order.
 ** @param n batch size.
 ** @param c number of channels.
 ** @param h height of input tensor.
 ** @param w width of input tensor.
 **/
__kernel
void chwn2nchw(__global float *chwn,
	__global float *nchw, int n, int c, int h, int w)
{
	int global_id = get_global_id(0);
	int batch_id = global_id / (c * h * w);
	if (batch_id < n) {
		int chw = global_id - batch_id * c * h * w;
		nchw[global_id] = chwn[chw * n + batch_id];
	}
}

__kernel
void winograd_convolution_f2x2_3x3(__global float *input,
	__global float *kernel, __global float *bias,
	__global float *output, int n, int c, int h, int w)
{
	
}