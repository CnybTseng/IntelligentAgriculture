#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

constant float4 G[8] = {
	(float4)(     1,       0,       0, 0),
	(float4)(-2/9.0,  -2/9.0,  -2/9.0, 0),
	(float4)(-2/9.0,   2/9.0,  -2/9.0, 0),
	(float4)(1/90.0,  1/45.0,  2/45.0, 0),
	(float4)(1/90.0, -1/45.0,  2/45.0, 0),
	(float4)(1/45.0,  1/90.0, 1/180.0, 0),
	(float4)(1/45.0, -1/90.0, 1/180.0, 0),
	(float4)(     0,       0,       1, 0)};

__kernel
void weight_transform_f6x6_3x3(global const float *weight,
	global float *transformed_weight, const int filter_channels, const int nfilters)
{
	int gx = get_global_id(0);
	
	float4 g[3];
	float4 Gg[8];
	float4 GgGT;
	
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		g[i] = vload4(0, weight + 9 * gx + i * 3);
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		Gg[i].x = G[i].x * g[0].x + G[i].y * g[1].x + G[i].z * g[2].x;
		Gg[i].y = G[i].x * g[0].y + G[i].y * g[1].y + G[i].z * g[2].y;
		Gg[i].z = G[i].x * g[0].z + G[i].y * g[1].z + G[i].z * g[2].z;
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		GgGT.x = Gg[i].x * G[0].x + Gg[i].y * G[0].y + Gg[i].z * G[0].z;
		GgGT.y = Gg[i].x * G[1].x + Gg[i].y * G[1].y + Gg[i].z * G[1].z;
		GgGT.z = Gg[i].x * G[2].x + Gg[i].y * G[2].y + Gg[i].z * G[2].z;
		GgGT.w = Gg[i].x * G[3].x + Gg[i].y * G[3].y + Gg[i].z * G[3].z;
		vstore4(GgGT, 0, transformed_weight + 64 * gx + i * 8);
		GgGT.x = Gg[i].x * G[4].x + Gg[i].y * G[4].y + Gg[i].z * G[4].z;
		GgGT.y = Gg[i].x * G[5].x + Gg[i].y * G[5].y + Gg[i].z * G[5].z;
		GgGT.z = Gg[i].x * G[6].x + Gg[i].y * G[6].y + Gg[i].z * G[6].z;
		GgGT.w = Gg[i].x * G[7].x + Gg[i].y * G[7].y + Gg[i].z * G[7].z;
		vstore4(GgGT, 0, transformed_weight + 64 * gx + i * 8 + 4);
	}
}

constant float4 BT[16] = {
	(float4)(1,      0, -21/4.0,       0), (float4)( 21/4.0,       0, -1, 0),
	(float4)(0,      1,       1, -17/4.0), (float4)(-17/4.0,       1,  1, 0),
	(float4)(0,     -1,       1,  17/4.0), (float4)(-17/4.0,      -1,  1, 0),
	(float4)(0,  1/2.0,   1/4.0,  -5/2.0), (float4)( -5/4.0,       2,  1, 0),
	(float4)(0, -1/2.0,   1/4.0,   5/2.0), (float4)( -5/4.0,      -2,  1, 0),
	(float4)(0,      2,       4,  -5/2.0), (float4)(     -5,   1/2.0,  1, 0),
	(float4)(0,     -2,       4,   5/2.0), (float4)(     -5,  -1/2.0,  1, 0),
	(float4)(0,     -1,       0,  21/4.0), (float4)(      0, -21/4.0,  0, 1)};

constant float B2[64] = {
	      1,       0,       0,      0,      0,      0,      0,       0,
	      0,       1,      -1,  1/2.0, -1/2.0,      2,     -2,      -1,
	-21/4.0,       1,       1,  1/4.0,  1/4.0,      4,      4,       0,
	      0, -17/4.0,  17/4.0, -5/2.0,  5/2.0, -5/2.0,  5/2.0,  21/4.0,
	 21/4.0, -17/4.0, -17/4.0, -5/4.0, -5/4.0,     -5,     -5,       0,
	      0,       1,      -1,      2,     -2,  1/2.0, -1/2.0, -21/4.0,
	     -1,       1,       1,      1,      1,      1,      1,       0,
	      0,       0,       0,      0,      0,      0,      0,       1
};

constant float4 AT[12] = {
	(float4)(1, 1,  1,  1), (float4)(  1, 32,  32, 0),
	(float4)(0, 1, -1,  2), (float4)( -2, 16, -16, 0),
	(float4)(0, 1,  1,  4), (float4)(  4,  8,   8, 0),
	(float4)(0, 1, -1,  8), (float4)( -8,  4,  -4, 0),
	(float4)(0, 1,  1, 16), (float4)( 16,  2,   2, 0),
	(float4)(0, 1, -1, 32), (float4)(-32,  1,  -1, 1)};	

__kernel
void input_transform_f6x6_3x3(__read_only image3d_t input,
	__write_only image2d_depth_t transformed_input)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	
	float4 d[16];
	float4 BTd[16];
	float4 BTdB[16];
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		d[(i << 1)]     = read_imagef(input, (int4)( gx << 1,      (gy << 3) + i, gz, 1));
		d[(i << 1) + 1] = read_imagef(input, (int4)((gx << 1) + 1, (gy << 3) + i, gz, 1));
		BTd[(i << 1)] = 0.0f;
		BTd[(i << 1) + 1] = 0.0f;
		BTdB[(i << 1)] = 0.0f;
		BTdB[(i << 1) + 1] = 0.0f;
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			BTd[(j << 1)]     += B2[(j << 3) + i] * d[(i << 1)];
			BTd[(j << 1) + 1] += B2[(j << 3) + i] * d[(i << 1) + 1];
		}
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		write_imagef(transformed_input, (int2)( gx << 1,      (gy << 3) + i), gz), BTdB[(i << 1)]);
		write_imagef(transformed_input, (int2)((gx << 1) + 1, (gy << 3) + i), gz), BTdB[(i << 1)]);
	}
}

__kernel
void winograd_convolution_f6x6_3x3(global const float *input,
	global const float *transformed_weight, global float *output)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
}