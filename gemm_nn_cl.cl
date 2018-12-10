__kernel void gemm_nn_cl(int m, int n, int k, float alpha, global float *A, int lda,
                         global float *B, int ldb, global float *C, int ldc)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x >= 0 && coord.x < n && coord.y >= 0 && coord.y < m)
	{
		for (int i = 0; i < k; ++i) {
			C[coord.y * ldc + coord.x] += alpha * A[coord.y * lda + i] * B[i * ldb + coord.x];
		}
	}
}