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