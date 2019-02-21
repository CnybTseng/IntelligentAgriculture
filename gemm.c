#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "gemm.h"

static void gemm_nn(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_tn(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_nt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_tt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);

#ifdef __INTEL_SSE__
static void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                        float *B, int ldb, float *C, int ldc) __attribute__((used));
#elif __ARM_NEON__
static void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                         float *B, int ldb, float *C, int ldc) __attribute__((used));
#endif
#ifdef OPENCL
extern cl_wrapper wrapper;
void gemm_nn_cl(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc);
void gemm_nn_cl_v1(int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc);
void gemm_nn_cl_v2(int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc);
#ifdef __linux__
void gemm_nn_cl_v3(int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc);
#endif
void gemm_nt_cl(int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float beta, float *C, int ldc);				
#endif					
					
/** @brief 通用的矩阵乘法,C=alhpa*A*B+beta*C.相乘前对A或B转置或共轭转置暂不支持.
 ** @param transa 转置(transa=1)或不转置(transa=0)矩阵A.
 ** @param transb 转置(transa=1)或不转置(transa=0)矩阵B.
 ** @param m 矩阵C的行数.
 ** @param n 矩阵C的列数.
 ** @param k 如果transa=0,k为矩阵A的列数.如果transa=1,k为矩阵A的行数.
 **          如果transb=0,k为矩阵B的行数.如果transb=1,k为矩阵B的列数.
 ** @param alpha 矩阵A和矩阵B乘积的标量乘子.
 ** @param A 矩阵A.
 ** @param lda 矩阵A或其转置的行步长.
 ** @param B 矩阵B.
 ** @param ldb 矩阵B或其转置的行步长.
 ** @param beta 矩阵C的标量乘子.
 ** @param C 矩阵C.
 ** @param ldc 矩阵C的行步长.
 **/
void gemm(int transa, int transb, int m, int n, int k, float alpha,
          float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
	const int mn = m * n;
#ifdef __ARM_NEON__
	const int batches = 4;
	const int excess = mn - mn % batches;
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t cs = vld1q_f32(C + i);
		cs = vmulq_n_f32(cs, beta);
		vst1q_f32(C + i, cs);
	}

	for (int i = excess; i < mn; ++i) {
		C[i] *= beta;
	}
#else
	#pragma omp parallel for
	for (int i = 0; i < mn; ++i) {
		C[i] *= beta;
	}
#endif
#endif	
	if (!transa && !transb) {
		gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else if (transa && !transb) {
		gemm_tn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else if (!transa && transb) {
		gemm_nt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else {
		gemm_tt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}

void gemm_nn(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
#ifdef __ARM_NEON__
	gemm_nn_neon(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#elif __INTEL_SSE__
	gemm_nn_sse(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#endif
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int l = 0; l < k; ++l) {
				sum += alpha * A[i * lda + l] * B[l * ldb + j];
			}
			C[i * ldc + j] += sum;
		}
	}
#else
	static double total = 0;
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	gemm_nn_cl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	gettimeofday(&t2, NULL);
	double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	total += duration;
	printf("gemm_nn: %f ms, total %f ms.\n", duration, total);
#endif
}
			 
void gemm_tn(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}		 
			 
void gemm_nt(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int l = 0; l < k; ++l) {
				sum += alpha * A[i * lda + l] * B[j * ldb + l];
			}
			C[i * ldc + j] += sum;
		}
	}
#else
	gemm_nt_cl(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}		 
			 
void gemm_tt(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#ifdef __INTEL_SSE__
void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                 float *B, int ldb, float *C, int ldc)
{
	
}

#elif __ARM_NEON__
void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                  float *B, int ldb, float *C, int ldc)
{
	
}
#endif	

#ifdef OPENCL
#if 0
void gemm_nn_cl(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc)
{	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_8x8", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel common_kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_common", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		m * k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		k * n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		m * n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, m * k * sizeof(float), 0, NULL, NULL, &errcode);
	// memcpy(h_A, A, m * k * sizeof(float));
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < k; ++x) {
			h_A[y * k + x] = A[y * k + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, k * n * sizeof(float), 0, NULL, NULL, &errcode);
	// memcpy(h_B, B, k * n * sizeof(float));
	for (int y = 0; y < k; ++y) {
		for (int x = 0; x < n; ++x) {
			h_B[y * n + x] = B[y * n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	// memcpy(h_C, C, m * n * sizeof(float));
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			h_C[y * n + x] = C[y * n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	errcode  = clSetKernelArg(common_kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(common_kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(common_kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(common_kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(common_kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(common_kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(common_kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(common_kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(common_kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(common_kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(common_kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	const int tile_rows = 8;
	const int tile_cols = 8;
	const int _m = (m / tile_rows) * tile_rows;
	const int _n = (n / tile_cols) * tile_cols;
	
	cl_event event;
	cl_uint work_dim = 2;
	
	if (_m && _n) {
		size_t global_work_size[] = {_n >> 3, _m >> 3};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
			NULL, 0, NULL, &event);
		
		if (n != _n) {
			size_t global_work_offset[] = {_n, 0};
			size_t global_work_size[] = {n - _n, _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, &event);
		}
		
		if (m != _m) {
			size_t global_work_offset[] = {0, _m};
			size_t global_work_size[] = {n, m - _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, &event);
		}
	} else {
		size_t global_work_size[] = {n, m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, NULL,
			global_work_size, NULL, 0, NULL, &event);
	}

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl(|%dx%d|*|%dx%d|): %f ms.\n", m, k, k, n, (end - start) * 1e-6f);
#endif

	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	// memcpy(C, h_C, m * n * sizeof(float));
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			C[y * n + x] = h_C[y * n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(common_kernel);
}
#else
void gemm_nn_cl(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc)
{	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_8x8", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}

	const int _m = ((m + 7) / 8) * 8;
	const int _n = ((n + 7) / 8) * 8;
	const int _k = ((k + 7) / 8) * 8;
	const int _lda = _k;
	const int _ldb = _n;
	const int _ldc = _n;
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		_m * _k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		_k * _n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		_m * _n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, _m * _k * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < k; ++x) {
			h_A[y * _k + x] = A[y * k + x];
		}
		for (int x = k; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, _k * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < k; ++y) {
		for (int x = 0; x < n; ++x) {
			h_B[y * _n + x] = B[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_B[y * _n + x] = 0;
		}
	}
	for (int y = k; y < _k; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_B[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, _m * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			h_C[y * _n + x] = C[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &_m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &_n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &_k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &_lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &_ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &_ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 3, _m >> 3};
	errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl(|%dx%d|*|%dx%d|=>|%dx%d|*|%dx%d|): %f ms.\n", m, k, k, n, _m, _k, _k, _n, (end - start) * 1e-6f);
#endif

	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, _m * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			C[y * n + x] = h_C[y * _n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
}
#endif

void gemm_nn_cl_v1(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc)
{	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_desktop_gpu", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel common_kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_common", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		m * k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		k * n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		m * n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, m * k * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_A, A, m * k * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, k * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_B, B, k * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_C, C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	errcode  = clSetKernelArg(common_kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(common_kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(common_kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(common_kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(common_kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(common_kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(common_kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(common_kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(common_kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(common_kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(common_kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	const int tile_rows = 8;
	const int tile_cols = 8;
	const int _m = (m / tile_rows) * tile_rows;
	const int _n = (n / tile_cols) * tile_cols;
	
	cl_event event;
	cl_uint work_dim = 2;
	
	if (_m && _n) {
		size_t global_work_size[] = {_n, _m};
		size_t local_work_size[]  = {16, 16};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
			local_work_size, 0, NULL, &event);
		
		if (n != _n) {
			size_t global_work_offset[] = {_n, 0};
			size_t global_work_size[] = {n - _n, _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, &event);
		}
		
		if (m != _m) {
			size_t global_work_offset[] = {0, _m};
			size_t global_work_size[] = {n, m - _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, &event);
		}
	} else {
		size_t global_work_size[] = {n, m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, NULL,
			global_work_size, NULL, 0, NULL, &event);
	}

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl_v1: %f ms.\n", (end - start) * 1e-6f);
#endif

	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(C, h_C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(common_kernel);
}

void gemm_nn_cl_v2(int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_8x4_f4", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	const int _m = ((m + 7) / 8) * 8;
	const int _n = ((n + 3) / 4) * 4;
	const int _k = ((k + 3) / 4) * 4;
	const int _lda = _k;
	const int _ldc = _n;

	cl_event event;
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		_m * _k * sizeof(float), NULL, &errcode);
	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _k * sizeof(float), 0, NULL, NULL, &errcode);

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < k; ++x) {
			h_A[y * _k + x] = A[y * k + x];
		}
		for (int x = k; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, &event);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		_m * _n * sizeof(float), NULL, &errcode);
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _n * sizeof(float), 0, NULL, NULL, &errcode);

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			h_C[y * _n + x] = C[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, &event);
		
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;	
	cl_image_format image_format = {CL_RGBA, CL_FLOAT};
	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = _n >> 2,
		.image_height = _k,
		.image_row_pitch = 0};
	cl_mem d_B = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	
	size_t origin[] = {0, 0, 0};
	size_t region[] = {_n >> 2, _k, 1};
	size_t image_row_pitch, image_slice_pitch;
	float *h_B = clEnqueueMapImage(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE, origin,
		region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);

	image_row_pitch = image_row_pitch >> 2;
	for (int y = 0; y < k; ++y) {
		for (int x = 0; x < n; ++x) {
			h_B[y * image_row_pitch + x] = B[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_B[y * image_row_pitch + x] = 0;
		}
	}
	for (int y = k; y < _k; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_B[y * image_row_pitch + x] = 0;
		}
	}

	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, &event);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &_m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &_n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &_k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &_lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 9, sizeof(int), &_ldc); 
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 2, _m >> 3};
	size_t local_work_size[] = {128, 8};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		local_work_size, 0, NULL, &event);

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl_v2(|%dx%d|*|%dx%d|=>|%dx%d|*|%dx%d|): %f ms.\n", m, k, k, n, _m, _k, _k, _n, (end - start) * 1e-6f);
#endif
		
	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _n * sizeof(float), 0, NULL, NULL, &errcode);	

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			C[y * n + x] = h_C[y * _n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, &event);
		
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
}

#ifdef __linux__
void gemm_nn_cl_v3(int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "matmul_8x4_blocks", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	
	
	
	
	cl_image_format image_a_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT};
	
	cl_image_desc image_a_desc;
	memset(&image_a_desc, 0, sizeof(cl_image_desc));
	image_a_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_a_desc.image_width = (k + 3) / 4,
	image_a_desc.image_height = ((m + 7) / 8) * 8,	
	image_a_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_a_format, image_a_desc);
	
	cl_ion_context ion_context_a = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_a_desc);
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_a = clCreateImage(wrapper.context, mem_flags, &image_a_format, &image_a_desc, &ion_context_a.ion_mem, &errcode);
		
	size_t image_a_origin[] = {0, 0, 0};
	size_t image_a_region[] = {image_a_desc.image_width, image_a_desc.image_height, 1};
	size_t image_a_row_pitch, image_a_slice_pitch;
	float *image_a_ptr = clEnqueueMapImage(wrapper.command_queue, image_a, CL_TRUE, CL_MAP_WRITE, image_a_origin,
		image_a_region, &image_a_row_pitch, &image_a_slice_pitch, 0, NULL, NULL, &errcode);

	image_a_row_pitch = image_a_row_pitch >> 2;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = *(A + i * k + j);
		}
		for (int j = k; j < image_a_desc.image_width; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = 0;
		}
	}
	for (int i = m; i < image_a_desc.image_height; ++i) {
		for (int j = 0; j < image_a_desc.image_width; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = 0;
		}
	}
	
	cl_event event;
	clEnqueueUnmapMemObject(wrapper.command_queue, image_a, image_a_ptr, 0, NULL, &event);
	
	
	
	
	
	cl_image_format image_b_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT};
	
	cl_image_desc image_b_desc;
	memset(&image_b_desc, 0, sizeof(cl_image_desc));
	image_b_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_b_desc.image_width = (n + 3) / 4,
	image_b_desc.image_height = ((k + 7) / 8) * 8,	
	image_b_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_b_format, image_b_desc);
	
	cl_ion_context ion_context_b = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_b_desc);
	
	mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_b = clCreateImage(wrapper.context, mem_flags, &image_b_format, &image_b_desc, &ion_context_b.ion_mem, &errcode);
	
	size_t image_b_origin[] = {0, 0, 0};
	size_t image_b_region[] = {image_b_desc.image_width, image_b_desc.image_height, 1};
	size_t image_b_row_pitch, image_b_slice_pitch;
	float *image_b_ptr = clEnqueueMapImage(wrapper.command_queue, image_b, CL_TRUE, CL_MAP_WRITE, image_b_origin,
		image_b_region, &image_b_row_pitch, &image_b_slice_pitch, 0, NULL, NULL, &errcode);

	image_b_row_pitch = image_b_row_pitch >> 2;
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = *(B + i * n + j);
		}
		for (int j = n; j < image_b_desc.image_width; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = 0;
		}
	}
	for (int i = k; i < image_b_desc.image_height; ++i) {
		for (int j = 0; j < image_b_desc.image_width; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = 0;
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, image_b, image_b_ptr, 0, NULL, &event);
	
	
	
	
	
	cl_image_format image_c_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT};
	
	cl_image_desc image_c_desc;
	memset(&image_c_desc, 0, sizeof(cl_image_desc));
	image_c_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_c_desc.image_width = (n + 3) / 4,
	image_c_desc.image_height = ((m + 7) / 8) * 8,	
	image_c_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_c_format, image_c_desc);
	
	cl_ion_context ion_context_c = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_c_desc);
	
	mem_flags = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_c = clCreateImage(wrapper.context, mem_flags, &image_c_format, &image_c_desc, &ion_context_c.ion_mem, &errcode);
	
	

	
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_a); 
	errcode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &image_b); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &image_c); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(cl_int), &k); 
	
	
	
	
	
	cl_uint work_dim = 2;
	size_t global_work_size[] = {image_b_desc.image_width, image_a_desc.image_height / 8};
	// size_t local_work_size[]  = {32, 16};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
	
	
	
	
#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl_v3: %f ms.\n", (end - start) * 1e-6f);
#endif	
	
	
	
	
	
	
	size_t image_c_origin[] = {0, 0, 0};
	size_t image_c_region[] = {image_c_desc.image_width, image_c_desc.image_height, 1};
	size_t image_c_row_pitch, image_c_slice_pitch;
	float *image_c_ptr = clEnqueueMapImage(wrapper.command_queue, image_c, CL_TRUE, CL_MAP_READ, image_c_origin,
		image_c_region, &image_c_row_pitch, &image_c_slice_pitch, 0, NULL, NULL, &errcode);

	image_c_row_pitch = image_c_row_pitch >> 2;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			*(C + i * n + j) = *(image_c_ptr + i * image_c_row_pitch + j);
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, image_a, image_a_ptr, 0, NULL, &event);
	
	
	
	
	
	cl_free_ion_context(wrapper, ion_context_a);
	cl_free_ion_context(wrapper, ion_context_b);
	cl_free_ion_context(wrapper, ion_context_c);
	clReleaseMemObject(image_a);
	clReleaseMemObject(image_b);
	clReleaseMemObject(image_c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
}
#endif

void gemm_nt_cl(int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nt_8x4", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel common_kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nt_common", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		m * k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		k * n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		m * n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, m * k * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_A, A, m * k * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, k * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_B, B, k * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_C, C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	errcode  = clSetKernelArg(common_kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(common_kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(common_kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(common_kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(common_kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(common_kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(common_kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(common_kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(common_kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(common_kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(common_kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	const int tile_rows = 8;
	const int tile_cols = 4;
	const int _m = (m / tile_rows) * tile_rows;
	const int _n = (n / tile_cols) * tile_cols;

	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 2, _m >> 3};
	errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL,
		global_work_size, NULL, 0, NULL, &event);
	
	if (n != _n) {
		size_t global_work_offset[] = {_n, 0};
		size_t global_work_size[] = {n - _n, _m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
			global_work_size, NULL, 0, NULL, &event);
	}
	
	if (m != _m) {
		size_t global_work_offset[] = {0, _m};
		size_t global_work_size[] = {n, m - _m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
			global_work_size, NULL, 0, NULL, &event);
	}

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nt_cl: %f ms.\n", (end - start) * 1e-6f);
#endif

	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(C, h_C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(common_kernel);
}
#endif	 