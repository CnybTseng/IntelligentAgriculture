#include <stdio.h>
#include <omp.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#ifdef OPENCL
#	include "cl_ez.h"
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
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			for (int l = 0; l < k; ++l) {
				C[i * ldc + j] += alpha * A[i * lda + l] * B[l * ldb + j];
			}
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
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
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
	cl_platform_layer *platform_layer = cl_create_platform_layer();
	cl_runtime *runtime = cl_create_runtime(platform_layer->devices[0], platform_layer->context, "gemm_nn_cl.cl");
	
	cl_int erret;
	cl_mem d_A = clCreateBuffer(platform_layer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		m * k * sizeof(float), A, &erret);
	cl_mem d_B = clCreateBuffer(platform_layer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		k * n * sizeof(float), B, &erret);
	cl_mem d_C = clCreateBuffer(platform_layer->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		m * n * sizeof(float), C, &erret);
	
	erret  = clSetKernelArg(runtime->kernel, 0, sizeof(int), &m);
	erret |= clSetKernelArg(runtime->kernel, 1, sizeof(int), &n); 
	erret |= clSetKernelArg(runtime->kernel, 2, sizeof(int), &k); 
	erret |= clSetKernelArg(runtime->kernel, 3, sizeof(float), &alpha);
	erret |= clSetKernelArg(runtime->kernel, 4, sizeof(cl_mem), d_A); 
	erret |= clSetKernelArg(runtime->kernel, 5, sizeof(int), &lda); 
	erret |= clSetKernelArg(runtime->kernel, 6, sizeof(cl_mem), d_B); 
	erret |= clSetKernelArg(runtime->kernel, 7, sizeof(int), &ldb); 
	erret |= clSetKernelArg(runtime->kernel, 8, sizeof(cl_mem), d_C); 
	erret |= clSetKernelArg(runtime->kernel, 9, sizeof(int), &ldc); 
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {m, n};
	erret = clEnqueueNDRangeKernel(runtime->cmdqueue, runtime->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
	
	clFinish(runtime->cmdqueue);
	
	cl_ulong start, end;
	erret  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	erret |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(event);
	
	printf("GPU: %f ms.\n", (end - start) * 1e-6f);
	
	erret = clEnqueueReadBuffer(runtime->cmdqueue, d_C, CL_FALSE, 0, m * n * sizeof(float), C, 0, NULL, NULL);
	
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	
	cl_destroy_runtime(runtime);
	cl_destroy_platform_layer(platform_layer);
}
#endif	 