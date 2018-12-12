#include <stdio.h>
#include <string.h>
#include <omp.h>
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
                    float *B, int ldb, float *C, int ldc);
static void gemm_tn(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);
static void gemm_nt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);
static void gemm_tt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);

#ifdef __INTEL_SSE__
static void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                        float *B, int ldb, float *C, int ldc);
#elif __ARM_NEON__
static void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                         float *B, int ldb, float *C, int ldc);
#endif
#ifdef OPENCL
void gemm_nn_cl(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float *C, int ldc);
void gemm_nt_cl(int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float *C, int ldc);				
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
	int mn = m * n;
#ifdef __ARM_NEON__
	int batches = 4;
	int excess = mn - mn % batches;
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
	if (!transa && !transb) {
		gemm_nn(m, n, k, alpha, A, lda, B, ldb, C, ldc);
	} else if (transa && !transb) {
		gemm_tn(m, n, k, alpha, A, lda, B, ldb, C, ldc);
	} else if (!transa && transb) {
		gemm_nt(m, n, k, alpha, A, lda, B, ldb, C, ldc);
	} else {
		gemm_tt(m, n, k, alpha, A, lda, B, ldb, C, ldc);
	}
}

void gemm_nn(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float *C, int ldc)
{
#ifdef OPENCL
	return gemm_nn_cl(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#endif
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int l = 0; l < k; ++l) {
				sum += alpha * A[i * lda + l] * B[l * ldb + j];
			}
			C[i * ldc + j] = sum;
		}
	}
}
			 
void gemm_tn(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}		 
			 
void gemm_nt(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float *C, int ldc)
{
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
}		 
			 
void gemm_tt(int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#ifdef __INTEL_SSE__
void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                 float *B, int ldb, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#elif __ARM_NEON__
void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                  float *B, int ldb, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}
#endif	

#ifdef OPENCL
void gemm_nn_cl(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float *C, int ldc)
{	
	cl_int errcode;
	cl_wrapper wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_program program = cl_make_wrapper_program(wrapper, "gemm.cl", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "gemm_nn_v1", &errcode);
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
	errcode |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 9, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {m, n};
	size_t local_work_size[] = {8, 8};
	errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		local_work_size, 0, NULL, &event);
	
	clFinish(wrapper.command_queue);

#ifdef CL_PROFILING_ENABLE	
	cl_ulong start, end;
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	printf("gemm_nn_cl: %f ms.\n", (end - start) * 1e-6f);
#endif
	
	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(C, h_C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	cl_destroy_wrapper(wrapper);
}

void gemm_nt_cl(int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float *C, int ldc)
{
	
}
#endif	 