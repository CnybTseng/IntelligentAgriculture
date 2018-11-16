#ifndef _GEMM_H_
#define _GEMM_H_

#ifdef __cplusplus
extern "C"
{
#endif

void gemm(int transa, int transb, int m, int n, int k, float alpha,
          float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif