#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel
void weight_transform_f6x6_3x3(global const float *weight, global const float *Gbuffer,
	global float *transformed_weight, const int filter_channels, const int nfilters, int xy_shift)
{
	int gx = get_global_id(0);
	
	float4 G[8];
	float4 g[3];
	float4 Gg[8];
	float4 GgGT;
	
	const int slice_pitch = (((filter_channels + 3) / 4) * 4) * (((nfilters + 3) / 4) * 4);
	const int xy = xy_shift + (gx / filter_channels) * (((filter_channels + 3) / 4) * 4) + gx % filter_channels;
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		G[i] = vload4(0, Gbuffer + (i << 2));
	}
	
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
		// vstore4(GgGT, 0, transformed_weight + 64 * gx + i * 8);
		
		transformed_weight[(i << 3) * slice_pitch + xy] = GgGT.x;
		transformed_weight[((i << 3) + 1) * slice_pitch + xy] = GgGT.y;
		transformed_weight[((i << 3) + 2) * slice_pitch + xy] = GgGT.z;
		transformed_weight[((i << 3) + 3) * slice_pitch + xy] = GgGT.w;
		
		GgGT.x = Gg[i].x * G[4].x + Gg[i].y * G[4].y + Gg[i].z * G[4].z;
		GgGT.y = Gg[i].x * G[5].x + Gg[i].y * G[5].y + Gg[i].z * G[5].z;
		GgGT.z = Gg[i].x * G[6].x + Gg[i].y * G[6].y + Gg[i].z * G[6].z;
		GgGT.w = Gg[i].x * G[7].x + Gg[i].y * G[7].y + Gg[i].z * G[7].z;
		// vstore4(GgGT, 0, transformed_weight + 64 * gx + i * 8 + 4);
		
		transformed_weight[((i << 3) + 4) * slice_pitch + xy] = GgGT.x;
		transformed_weight[((i << 3) + 5) * slice_pitch + xy] = GgGT.y;
		transformed_weight[((i << 3) + 6) * slice_pitch + xy] = GgGT.z;
		transformed_weight[((i << 3) + 7) * slice_pitch + xy] = GgGT.w;
	}
}

__kernel
void input_transform_f6x6_3x3(__read_only image2d_t input, const int height_per_channel,
	global const float *BTbuffer, global float *transformed_input)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	
	float4 d[16];
	float4 BT[16];
	float4 BTd[16];
	float4 BTdB;
	
	const int slice_pitch = (((get_global_size(0) * get_global_size(1) + 3) / 4) * 4) * (((get_global_size(2) + 3) / 4) * 4);
	const int xy = gz * (((get_global_size(0) * get_global_size(1) + 3) / 4) * 4) + gy * get_global_size(0) + gx;
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		d[i << 1] = read_imagef(input, (int2)(gx << 1, gz * height_per_channel + gy * 6 + i));
		d[(i << 1) + 1] = read_imagef(input, (int2)((gx << 1) + 1, gz * height_per_channel + gy * 6 + i));
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		BT[i << 1] = vload4(0, BTbuffer + (i << 3));
		BT[(i << 1) + 1] = vload4(0, BTbuffer + (i << 3) + 4);
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {		
		BTd[i << 1] = BT[i << 1].x * d[0] + BT[i << 1].y * d[2] + BT[i << 1].z * d[4] + BT[i << 1].w * d[6] +
			BT[(i << 1) + 1].x * d[8] + BT[(i << 1) + 1].y * d[10] + BT[(i << 1) + 1].z * d[12] + BT[(i << 1) + 1].w * d[14];
		
		BTd[(i << 1) + 1] = BT[i << 1].x * d[1] + BT[i << 1].y * d[3] + BT[i << 1].z * d[5] + BT[i << 1].w * d[7] +
			BT[(i << 1) + 1].x * d[9] + BT[(i << 1) + 1].y * d[11] + BT[(i << 1) + 1].z * d[13] + BT[(i << 1) + 1].w * d[15];
	}
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		BTdB.x = dot(BTd[i << 1], BT[0]) + dot(BTd[(i << 1) + 1], BT[1]);
		BTdB.y = dot(BTd[i << 1], BT[2]) + dot(BTd[(i << 1) + 1], BT[3]);
		BTdB.z = dot(BTd[i << 1], BT[4]) + dot(BTd[(i << 1) + 1], BT[5]);
		BTdB.w = dot(BTd[i << 1], BT[6]) + dot(BTd[(i << 1) + 1], BT[7]);
		
		transformed_input[(i << 3) * slice_pitch + xy] = BTdB.x;
		transformed_input[((i << 3) + 1) * slice_pitch + xy] = BTdB.y;
		transformed_input[((i << 3) + 2) * slice_pitch + xy] = BTdB.z;
		transformed_input[((i << 3) + 3) * slice_pitch + xy] = BTdB.w;
		
		BTdB.x = dot(BTd[i << 1], BT[8]) + dot(BTd[(i << 1) + 1], BT[9]);
		BTdB.y = dot(BTd[i << 1], BT[10]) + dot(BTd[(i << 1) + 1], BT[11]);
		BTdB.z = dot(BTd[i << 1], BT[12]) + dot(BTd[(i << 1) + 1], BT[13]);
		BTdB.w = dot(BTd[i << 1], BT[14]) + dot(BTd[(i << 1) + 1], BT[15]);
		
		transformed_input[((i << 3) + 4) * slice_pitch + xy] = BTdB.x;
		transformed_input[((i << 3) + 5) * slice_pitch + xy] = BTdB.y;
		transformed_input[((i << 3) + 6) * slice_pitch + xy] = BTdB.z;
		transformed_input[((i << 3) + 7) * slice_pitch + xy] = BTdB.w;
	}
}

__kernel
void matrix_multiply(global const float *transformed_weight, global const float *transformed_input,
	global float *output, const int round_filter_channels, const int round_num_filters, const int tile_size,
	const int round_num_tiles)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	
	float4 a[4];
	float4 b[4];
	float4 c[4];
	
	const int slice_pitch = tile_size * tile_size * round_num_tiles;
	
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		c[i] = 0;
	}
	
	for (int i = 0; i < round_filter_channels; i += 4) {
		#pragma unroll
		for (int j = 0; j < 4; j++) {
			a[j] = vload4(0, transformed_weight + gz * round_filter_channels * round_num_filters +
				((gy << 2) + j) * round_filter_channels + i);
		}
		
		#pragma unroll
		for (int j = 0; j < 4; j++) {
			b[j] = vload4(0, transformed_input + gz * round_num_tiles * round_filter_channels +
				(i + j) * round_num_tiles + (gx << 2));
		}
		
		#pragma unroll
		for (int j = 0; j < 4; j++) {
			c[j] += a[j].x * b[0] + a[j].y * b[1] + a[j].z * b[2] + a[j].w * b[3];
		}
	}
	
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		// vstore4(c[i], 0, output + gz * round_num_tiles * round_num_filters + ((gy << 2) + i) * round_num_tiles + (gx << 2));
		output[((gy << 2) + i) * slice_pitch + ((gx << 2) + 0) * tile_size * tile_size + gz] = c[i].x;
		output[((gy << 2) + i) * slice_pitch + ((gx << 2) + 1) * tile_size * tile_size + gz] = c[i].y;
		output[((gy << 2) + i) * slice_pitch + ((gx << 2) + 2) * tile_size * tile_size + gz] = c[i].z;
		output[((gy << 2) + i) * slice_pitch + ((gx << 2) + 3) * tile_size * tile_size + gz] = c[i].w;
	}
}

__kernel
void inverse_output_transform_f6x6_3x3(global const float *output, global float *inverse_transformed_output,
	global const float *ATBuffer, const int round_num_tiles, const int tile_size, const int round_num_filters,
	const int valid_width, const int valid_height, const int ntilesX, const int ntilesY)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	float4 o[16];
	float4 AT[12];
	float4 ATo[12];
	float4 AToA;
	
	const int output_slice_pitch = round_num_tiles * tile_size * tile_size;
	const int inverse_output_slice_pitch = ntilesX * ntilesY * 36;

	#pragma unroll
	for (int i = 0; i < 16; i++) {
		o[i] = vload4(0, output + gy * output_slice_pitch + gx * tile_size * tile_size + (i << 2));
	}
	
	#pragma unroll
	for (int i = 0; i < 12; i++) {
		AT[i] = vload4(0, ATBuffer + (i << 2));
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		ATo[i << 1] = AT[i << 1].x * o[0] + AT[i << 1].y * o[2] + AT[i << 1].z * o[4] + AT[i << 1].w * o[6] +
			AT[(i << 1) + 1].x * o[8] + AT[(i << 1) + 1].y * o[10] + AT[(i << 1) + 1].z * o[12] + AT[(i << 1) + 1].w * o[14];
		ATo[(i << 1) + 1] = AT[i << 1].x * o[1] + AT[i << 1].y * o[3] + AT[i << 1].z * o[5] + AT[i << 1].w * o[7] +
			AT[(i << 1) + 1].x * o[9] + AT[(i << 1) + 1].y * o[11] + AT[(i << 1) + 1].z * o[13] + AT[(i << 1) + 1].w * o[15];
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		AToA.x = dot(ATo[i << 1], AT[0]) + dot(ATo[(i << 1) + 1], AT[1]);
		AToA.y = dot(ATo[i << 1], AT[2]) + dot(ATo[(i << 1) + 1], AT[3]);
		AToA.z = dot(ATo[i << 1], AT[4]) + dot(ATo[(i << 1) + 1], AT[5]);
		AToA.w = dot(ATo[i << 1], AT[6]) + dot(ATo[(i << 1) + 1], AT[7]);
		
		vstore4(AToA, 0, inverse_transformed_output + gy * inverse_output_slice_pitch + ((gx / ntilesX) * 6 + i) *
			(ntilesX * 6) + ((gx % ntilesX) * 6));
		
		AToA.x = dot(ATo[i << 1], AT[8]) + dot(ATo[(i << 1) + 1], AT[9]);
		AToA.y = dot(ATo[i << 1], AT[10]) + dot(ATo[(i << 1) + 1], AT[11]);
		
		vstore2(AToA.xy, 0, inverse_transformed_output + gy * inverse_output_slice_pitch + ((gx / ntilesX) * 6 + i) *
			(ntilesX * 6) + ((gx % ntilesX) * 6) + 4);
	}
}