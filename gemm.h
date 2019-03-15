#ifndef _GEMM_H_
#define _GEMM_H_

#ifdef __cplusplus
extern "C"
{
#endif

struct gemm_context;
typedef struct gemm_context gemm_context;

gemm_context *create_gemm_context(int transa, int transb, int m, int n, int k);
void gemm(gemm_context *context, int transa, int transb, int m, int n, int k, float alpha,
          float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);
void free_gemm_context(gemm_context *context);
#ifdef __cplusplus
}
#endif

#endif