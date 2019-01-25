/** @name Single precision floating General Matrix Multiply. Not transpose matrix A and B.
 ** @ { */
__kernel void
sgemm_nn_common(int m, int n, int k, float alpha, __global float *A, int lda,
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
sgemm_nn_8x4(int m, int n, int k, float alpha, __global float *A, int lda,
             __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);

	enum {TILE_HEIGHT = 8};
	float  a[TILE_HEIGHT];
	// float4 a[TILE_HEIGHT];
	float4 b;
	// float4 b[4];
	float4 c[TILE_HEIGHT];
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload4(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
	
	for (int i = 0; i < k; ++i) {
	// for (int i = 0; i < k; i += 4) {
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
			// a[j] = vload4(0, A + ((tile_row << 3) + j) * lda + i);
		}
		
		b = vload4(0, B + i * ldb + (tile_col << 2));
		// #pragma unroll
		// for (int j = 0; j < 4; j++) {
		// 	b[j] = vload4(0, B + (i + j) * ldb + (tile_col << 2));
		// }
		
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
			// c[j] += alpha * (a[j].x * b[0] + a[j].y * b[1] + a[j].z * b[2] + a[j].w * b[3]);
		}
	}
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore4(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
}

__kernel void
sgemm_nn_8x8(int m, int n, int k, float alpha, __global float *A, int lda,
             __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	enum {TILE_HEIGHT = 8};
	float  a[TILE_HEIGHT];
	float8 b;
	float8 c[TILE_HEIGHT];
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload8(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
		}
		
		b = vload8(0, B + i * ldb + (tile_col << 3));
		
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore8(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 3));
	}
}

__kernel void
sgemm_nn_8x16(int m, int n, int k, float alpha, __global float *A, int lda,
              __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	enum {TILE_HEIGHT = 8};
	float   a[TILE_HEIGHT];
	float16 b;
	float16 c[TILE_HEIGHT];
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload16(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 4));
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = A[((tile_row << 3) + j) * lda + i];
		}
		
		b = vload16(0, B + i * ldb + (tile_col << 4));
		
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		vstore16(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 4));
	}
}

__kernel void
sgemm_nn_8x4_f4(const int m, const int n, const int k, const float alpha, __global const float *A, const int lda,
                __read_only image2d_t B, const float beta, __global float *C, const int ldc)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	float4 a[8];
	float4 b[4];
	float4 c[8];
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		c[i] = beta * vload4(0, C + ((gy << 3) + i) * ldc + (gx << 2));
	}
	
	for (int pos = 0; pos < k; pos += 4) {
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			a[i] = vload4(0, A + ((gy << 3) + i) * lda + pos);
		}
		
		#pragma unroll
		for (int i = 0; i < 4; ++i) {
			b[i] = read_imagef(B, (int2)(gx, pos + i));
		}
		
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			c[i] += alpha * (a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3]);
		}
	}
	
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		vstore4(c[i], 0, C + ((gy << 3) + i) * ldc + (gx << 2));
	}
}

__kernel void
sgemm_nn_desktop_gpu(int m, int n, int k, float alpha, __global float *A, int lda,
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
		
		#pragma unroll
		for (int j = 0; j < tile_size; ++j) {
			acc += alpha * tile_A[local_row][j] * tile_B[j][local_col];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	float c = beta * C[global_row * ldc + global_col];
	C[global_row * ldc + global_col] = acc + c;
}

/** @ }*/
/** @name Single precision floating General Matrix Multiply.
 ** @ { */

__kernel void
sgemm_nt_common(int m, int n, int k, float alpha, __global float *A, int lda,
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

__kernel void
sgemm_nt_1x4(int m, int n, int k, float alpha, __global float *A, int lda,
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
sgemm_nt_8x4(int m, int n, int k, float alpha, __global float *A, int lda,
             __global float *B, int ldb, float beta, __global float *C, int ldc)
{
	const int tile_col = get_global_id(0);
	const int tile_row = get_global_id(1);
	
	enum {TILE_HEIGHT = 8};
	float4 a[TILE_HEIGHT];
	float4 b[4];
	float4 c84[TILE_HEIGHT][4] = {
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
	float4 c[TILE_HEIGHT] = {0, 0, 0, 0};
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j] = beta * vload4(0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
	
	for (int i = 0; i < k; i += 4) {
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; ++j) {
			a[j] = vload4(0, A + ((tile_row << 3) + j) * lda + i);
		}
		
		#pragma unroll
		for (int j = 0; j < 4; ++j) {
			b[j] = vload4(0, B + ((tile_col << 2) + j) * ldb + i);
		}
		
		#pragma unroll
		for (int j = 0; j < TILE_HEIGHT; j++) {
			c84[j][0] += alpha * a[j] * b[0];
			c84[j][1] += alpha * a[j] * b[1];
			c84[j][2] += alpha * a[j] * b[2];
			c84[j][3] += alpha * a[j] * b[3];
		}
	}
	
	#pragma unroll
	for (int j = 0; j < TILE_HEIGHT; ++j) {
		c[j].x += c84[j][0].x + c84[j][0].y + c84[j][0].z + c84[j][0].w;
		c[j].y += c84[j][1].x + c84[j][1].y + c84[j][1].z + c84[j][1].w;
		c[j].z += c84[j][2].x + c84[j][2].y + c84[j][2].z + c84[j][2].w;
		c[j].w += c84[j][3].x + c84[j][3].y + c84[j][3].z + c84[j][3].w;
		
		vstore4(c[j], 0, C + ((tile_row << 3) + j) * ldc + (tile_col << 2));
	}
}

/** @ }*/