#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

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

constant float4 a[8] = {
	(float4)(     1,       0,       0, 0),
	(float4)(-2/9.0,  -2/9.0,  -2/9.0, 0),
	(float4)(-2/9.0,   2/9.0,  -2/9.0, 0),
	(float4)(1/90.0,  1/45.0,  2/45.0, 0),
	(float4)(1/90.0, -1/45.0,  2/45.0, 0),
	(float4)(1/45.0,  1/90.0, 1/180.0, 0),
	(float4)(1/45.0, -1/90.0, 1/180.0, 0),
	(float4)(     0,       0,       1, 0)};

/** @brief Weight matrix transformation for Winograd convolution F(6x6,3x3).
 ** @param weight weight tensor 3D-image.
 ** @param G transformation matrix.
 ** @param transformed_weight transfromed weight tensor 3D-image.
 **/
__kernel
void weight_transform_f6x6_3x3(__read_only image3d_t weight,
	__write_only image3d_t transformed_weight)
{
	int gx = get_global_id(0);	// == 0
	int gy = get_global_id(1);	// == 0
	int gz = get_global_id(2);	// == [0,filter channel * number of filters)
	
	float4 b[4];
	float4 c[8];
	float4 d1, d2;
	int pos = 0;
	float zero = 0;
	
	// load 4x4 weight matrix.
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		b[i] = read_imagef(weight, (int4)(gx, gy + i, gz, 0));
	}
	
	// calculate 8x4 G*g.
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		c[i].x = a[i].x * b[0].x + a[i].y * b[1].x + a[i].z * b[2].x + a[i].w * b[3].x;
		c[i].y = a[i].x * b[0].y + a[i].y * b[1].y + a[i].z * b[2].y + a[i].w * b[3].y;
		c[i].z = a[i].x * b[0].z + a[i].y * b[1].z + a[i].z * b[2].z + a[i].w * b[3].z;
		c[i].w = a[i].x * b[0].w + a[i].y * b[1].w + a[i].z * b[2].w + a[i].w * b[3].w;
	}
	
	// calculate 8x8 G*g*Gt.
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		d1.x = c[i].x * a[0].x + c[i].y * a[0].y + c[i].z * a[0].z + c[i].w * a[0].w;
		d1.y = c[i].x * a[1].x + c[i].y * a[1].y + c[i].z * a[1].z + c[i].w * a[1].w;
		d1.z = c[i].x * a[2].x + c[i].y * a[2].y + c[i].z * a[2].z + c[i].w * a[2].w;
		d1.w = c[i].x * a[3].x + c[i].y * a[3].y + c[i].z * a[3].z + c[i].w * a[3].w;
		write_imagef(transformed_weight, (int4)(gx, gy + i, gz, 0), d1);
		d2.x = c[i].x * a[4].x + c[i].y * a[4].y + c[i].z * a[4].z + c[i].w * a[4].w;
		d2.y = c[i].x * a[5].x + c[i].y * a[5].y + c[i].z * a[5].z + c[i].w * a[5].w;
		d2.z = c[i].x * a[6].x + c[i].y * a[6].y + c[i].z * a[6].z + c[i].w * a[6].w;
		d2.w = c[i].x * a[7].x + c[i].y * a[7].y + c[i].z * a[7].z + c[i].w * a[7].w;
		write_imagef(transformed_weight, (int4)(gx + 1, gy + i, gz, 0), d2);
	}
}