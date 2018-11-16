#include <stdio.h>
#include "gemm.h"

static void gemm_nn(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);
static void gemm_tn(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);
static void gemm_nt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);
static void gemm_tt(int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float *C, int ldc);

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
	for (int i = 0; i < mn; ++i) {
		C[i] *= beta;
	}
	
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