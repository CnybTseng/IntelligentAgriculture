__kernel void
gemm_nn_common(int m, int n, int k, float alpha, __global float *A, int lda,
               __global float *B, int ldb, float beta, __global float *C, int ldc)
{	
	const int global_col = get_global_id(0);
	const int global_row = get_global_id(1);
	
	float sum = beta * C[global_row * ldc + global_col];
	for (int l = 0; l < k; ++l) {
		sum += alpha * A[global_row * lda + l] * B[l * ldb + global_col];
	}
	
	C[global_row * ldc + global_col] = sum;
}

__kernel void
gemm_nt_common(int m, int n, int k, float alpha, __global float *A, int lda,
               __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int global_col = get_global_id(0);
	const int global_row = get_global_id(1);
	
	float sum = beta * C[global_row * ldc + global_col];
	for (int l = 0; l < k; ++l) {
		sum += alpha * A[global_row * lda + l] * B[global_col * ldb + l];
	}
	
	C[global_row * ldc + global_col] = sum;
}

#define TILE_HEIGHT 8

__kernel void
gemm_nn_8x4(int m, int n, int k, float alpha, __global float *A, int lda,
            __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float  a[TILE_HEIGHT];
	float4 b;
	float4 c[TILE_HEIGHT];
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload4(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
		}
		
		b = vload4(0, B + i * ldb + (tile_col << 2));
		
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore4(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
}

__kernel void
gemm_nn_8x8(int m, int n, int k, float alpha, __global float *A, int lda,
            __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float  a[TILE_HEIGHT];
	float8 b;
	float8 c[TILE_HEIGHT];
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload8(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
		}
		
		b = vload8(0, B + i * ldb + (tile_col << 3));
		
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore8(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
}

__kernel void
gemm_nn_8x16(int m, int n, int k, float alpha, __global float *A, int lda,
             __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float   a[TILE_HEIGHT];
	float16 b;
	float16 c[TILE_HEIGHT];
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload16(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 4));
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
		}
		
		b = vload16(0, B + i * ldb + (tile_col << 4));
		
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore16(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 4));
	}
}

__kernel void
gemm_nt_1x4(int m, int n, int k, float alpha, __global float *A, int lda,
            __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float4 a;
	float4 b;
	float4 c4[4] = {0, 0, 0, 0};
	float4 c = {0, 0, 0, 0};
	
	c = beta * vload4(0, C + tile_row * ldc + (tile_col << 2));
	
	for (int i = 0; i < k; i += 4) {
		a = vload4(0, A + tile_row * lda + i);
		for (int j = 0; j < 4; ++j) {
			b = vload4(0, B + ((tile_col << 2) + j) * ldb + i);
			c4[j] += alpha * a * b;
		}
	}
	
	c.x += c4[0].x + c4[0].y + c4[0].z + c4[0].w;
	c.y += c4[1].x + c4[1].y + c4[1].z + c4[1].w;
	c.z += c4[2].x + c4[2].y + c4[2].z + c4[2].w;
	c.w += c4[3].x + c4[3].y + c4[3].z + c4[3].w;
	
	vstore4(c, 0, C + tile_row * ldc + (tile_col << 2));
}

__kernel void
gemm_nt_8x4(int m, int n, int k, float alpha, __global float *A, int lda,
            __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float4 a[TILE_HEIGHT];
	float4 b[4];
	float4 c84[TILE_HEIGHT][4] = {
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
	float4 c[TILE_HEIGHT] = {0, 0, 0, 0};
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload4(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
	
	for (int i = 0; i < k; i += 4) {
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = vload4(0, A + ((tile_row << 3) + j) * lda + i);
		}
		
		#pragma unroll 4
		for (int j = 0; j < 4; ++j) {
			b[j] = vload4(0, B + ((tile_col << 2) + j) * ldb + i);
		}
		
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; j++) {
			c84[j][0] += alpha * a[j] * b[0];
			c84[j][1] += alpha * a[j] * b[1];
			c84[j][2] += alpha * a[j] * b[2];
			c84[j][3] += alpha * a[j] * b[3];
		}
	}
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j].x += c84[j][0].x + c84[j][0].y + c84[j][0].z + c84[j][0].w;
		c[j].y += c84[j][1].x + c84[j][1].y + c84[j][1].z + c84[j][1].w;
		c[j].z += c84[j][2].x + c84[j][2].y + c84[j][2].z + c84[j][2].w;
		c[j].w += c84[j][3].x + c84[j][3].y + c84[j][3].z + c84[j][3].w;
		
		// c84[j][0].xy += c84[j][0].zw;
		// c[j].x += c84[j][0].x + c84[j][0].y;
		// 
		// c84[j][1].xy += c84[j][1].zw;
		// c[j].y += c84[j][1].x + c84[j][1].y;
		// 
		// c84[j][2].xy += c84[j][2].zw;
		// c[j].z += c84[j][2].x + c84[j][2].y;
		// 
		// c84[j][3].xy += c84[j][3].zw;
		// c[j].w += c84[j][3].x + c84[j][3].y;
		
		vstore4(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
}

__kernel void
gemm_nt_8x8(int m, int n, int k, float alpha, __global float *A, int lda,
            __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	float8 a[TILE_HEIGHT];
	float8 b[8];
	float8 c88[TILE_HEIGHT][8];
	float8 c[TILE_HEIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c88[j][0] = 0;
		c88[j][1] = 0;
		c88[j][2] = 0;
		c88[j][3] = 0;
		c88[j][4] = 0;
		c88[j][5] = 0;
		c88[j][6] = 0;
		c88[j][7] = 0;
		c[j] = beta * vload8(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
	
	for (int i = 0; i < k; i += 8) {
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = vload8(0, A + ((tile_row << 3) + j) * lda + i);
		}
		
		#pragma unroll 8
		for (int j = 0; j < 8; ++j) {
			b[j] = vload8(0, B + ((tile_col << 3) + j) * ldb + i);
		}
		
		#pragma unroll TILE_HEIGHT
		for (int j = 0; j < TILE_HEIGHT; j++) {
			c88[j][0] += alpha * a[j] * b[0];
			c88[j][1] += alpha * a[j] * b[1];
			c88[j][2] += alpha * a[j] * b[2];
			c88[j][3] += alpha * a[j] * b[3];
			c88[j][4] += alpha * a[j] * b[4];
			c88[j][5] += alpha * a[j] * b[5];
			c88[j][6] += alpha * a[j] * b[6];
			c88[j][7] += alpha * a[j] * b[7];
		}
	}
	
	#pragma unroll TILE_HEIGHT
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j].s0 += c88[j][0].s0 + c88[j][0].s1 + c88[j][0].s2 + c88[j][0].s3 + c88[j][0].s4 + c88[j][0].s5 + c88[j][0].s6 + c88[j][0].s7;
		c[j].s1 += c88[j][1].s0 + c88[j][1].s1 + c88[j][1].s2 + c88[j][1].s3 + c88[j][1].s4 + c88[j][1].s5 + c88[j][1].s6 + c88[j][1].s7;
		c[j].s2 += c88[j][2].s0 + c88[j][2].s1 + c88[j][2].s2 + c88[j][2].s3 + c88[j][2].s4 + c88[j][2].s5 + c88[j][2].s6 + c88[j][2].s7;
		c[j].s3 += c88[j][3].s0 + c88[j][3].s1 + c88[j][3].s2 + c88[j][3].s3 + c88[j][3].s4 + c88[j][3].s5 + c88[j][3].s6 + c88[j][3].s7;
		c[j].s4 += c88[j][4].s0 + c88[j][4].s1 + c88[j][4].s2 + c88[j][4].s3 + c88[j][4].s4 + c88[j][4].s5 + c88[j][4].s6 + c88[j][4].s7;
		c[j].s5 += c88[j][5].s0 + c88[j][5].s1 + c88[j][5].s2 + c88[j][5].s3 + c88[j][5].s4 + c88[j][5].s5 + c88[j][5].s6 + c88[j][5].s7;
		c[j].s6 += c88[j][6].s0 + c88[j][6].s1 + c88[j][6].s2 + c88[j][6].s3 + c88[j][6].s4 + c88[j][6].s5 + c88[j][6].s6 + c88[j][6].s7;
		c[j].s7 += c88[j][7].s0 + c88[j][7].s1 + c88[j][7].s2 + c88[j][7].s3 + c88[j][7].s4 + c88[j][7].s5 + c88[j][7].s6 + c88[j][7].s7;
		vstore8(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
}

__kernel void
gemm_nn_v2(int m, int n, int k, float alpha, __global float *A, int lda,
        __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int local_row = get_local_id(0);
	const int local_col = get_local_id(1);

	enum {tile_size = 16};
	__local float tile_A[tile_size][tile_size];
	__local float tile_B[tile_size][tile_size];

	const int global_row = get_group_id(0) * tile_size + local_row;
	const int global_col = get_group_id(1) * tile_size + local_col;
	
	float acc = 0;
	const int ntiles = k / tile_size;
	for (int i = 0; i < ntiles; ++i) {
		tile_A[local_row][local_col] = A[global_row * lda + i * tile_size + local_col];
		tile_B[local_row][local_col] = B[(i * tile_size + local_row) * ldb + global_col];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int j = 0; j < tile_size; ++j) {
			acc += alpha * tile_A[local_row][j] * tile_B[j][local_col];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	C[global_row * ldc + global_col] += acc;
}

__kernel void
gemm_nn_v3(int m, int n, int k, float alpha, __global float *A, int lda,
           __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int local_row = get_local_id(0);
	const int local_col = get_local_id(1);

	enum {tile_size = 16};
	__local float tile_A[tile_size][tile_size];
	__local float tile_B[tile_size][tile_size];

	const int global_row = get_group_id(0) * tile_size + local_row;
	const int global_col = get_group_id(1) * tile_size + local_col;
	
	float acc = 0;
	const int ntiles = k / tile_size + (k % tile_size != 0);
	for (int i = 0; i < ntiles; ++i) {
		int ax = i * tile_size + local_col;
		if (global_row < m && ax < k)
			tile_A[local_row][local_col] = A[global_row * lda + ax];
		else
			tile_A[local_row][local_col] = 0;
		
		int by = i * tile_size + local_row;
		if (by < k && global_col < n)
			tile_B[local_row][local_col] = B[by * ldb + global_col];
		else
			tile_B[local_row][local_col] = 0;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int j = 0; j < tile_size; ++j) {
			acc += alpha * tile_A[local_row][j] * tile_B[j][local_col];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (global_row < m && global_col < n)
		C[global_row * ldc + global_col] += acc;
}

__kernel void matmul_8x4_blocks(__global const float *matrix_a,
                                __global const float *matrix_b,
                                __global       float *matrix_c,
                                               int    matrix_b_width,
                                               int    matrix_a_width)
{
    const int wid_x = get_global_id(0);
    const int wid_y = get_global_id(1);

    float  a[8];
    float4 b;
    float4 c[8];

    for (int i = 0; i < 8; ++i)
    {
        c[i] = (float4)(0.0f);
    }

    for (int j = 0; j < matrix_a_width; ++j)
    {
        b = vload4(0, matrix_b + j * matrix_b_width + (wid_x * 4));

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            a[i] = matrix_a[((wid_y * 8) + i) * matrix_a_width + j];
        }

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            c[i] += a[i] * b;
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        vstore4(c[i], 0, matrix_c + ((wid_y * 8) + i) * matrix_b_width + (wid_x * 4));
    }
}

__kernel void matmul_remainder(__global const  float *matrix_a,
                               __global const  float *matrix_b,
                               __global        float *matrix_c,
                                               int    x_rem_start,
                                               int    y_rem_start,
                                               int    matrix_b_width,
                                               int    matrix_a_width)
{
    const int wid_x = get_global_id(0) + x_rem_start;
    const int wid_y = get_global_id(1) + y_rem_start;

    float c     = 0.0f;
    int   a_idx = matrix_a_width * wid_y;
    int   b_idx = wid_x;

#pragma unroll 8
    for (int i = 0; i < matrix_a_width; ++i)
    {
        c += matrix_a[a_idx] * matrix_b[b_idx];
        ++a_idx;
        b_idx += matrix_b_width;
    }

    const int c_idx = wid_x + matrix_b_width * wid_y;
    matrix_c[c_idx] = c;
}