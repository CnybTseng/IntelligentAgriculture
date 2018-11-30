#include <math.h>
#include <omp.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#include "activation.h"

static void relu_activate(float *X, int n);
static void leaky_activate(float *X, int n);
static void linear_activate(float *X, int n);
static void logistic_active(float *X, int n);

#ifdef __INTEL_SSE__
static void relu_activate_sse(float *X, int n);
static void leaky_activate_sse(float *X, int n);
static void logistic_active_sse(float *X, int n);
#elif __ARM_NEON__
static void relu_activate_neon(float *X, int n);
static void leaky_activate_neon(float *X, int n);
static void logistic_active_neon(float *X, int n);
#endif

void activate(float *X, int n, ACTIVATION activation)
{
	if (activation == RELU) {
		relu_activate(X, n);
	} else if (activation == LEAKY) {
		leaky_activate(X, n);
	} else if (activation == LINEAR){
		linear_activate(X, n);
	} else if (activation == LOGISTIC) {
		logistic_active(X, n);
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void relu_activate(float *X, int n)
{
#ifdef __ARM_NEON__	
	return relu_activate_neon(X, n);
#endif
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		X[i] = (X[i] > 0) * X[i];
	}
}

void leaky_activate(float *X, int n)
{
#ifdef __ARM_NEON__
	return leaky_activate_neon(X, n);
#endif
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		X[i] = (X[i] > 0) ? X[i] : 0.1 * X[i];
	}
}

void linear_activate(float *X, int n)
{
	return;
}

void logistic_active(float *X, int n)
{
#ifdef __ARM_NEON__
	return logistic_active_neon(X, n);
#endif
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		X[i] = 1 / (1 + exp(-X[i]));
	}
}

#ifdef __INTEL_SSE__
void relu_activate_sse(float *X, int n)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void leaky_activate_sse(float *X, int n)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void logistic_active_sse(float *X, int n)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#elif __ARM_NEON__
void relu_activate_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t zeros = vdupq_n_f32(0);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		uint32x4_t mask = vcgtq_f32(xs, zeros);
		float32x4_t zs = vbslq_f32(mask, xs, zeros);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) * X[i];
	}
}

void leaky_activate_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t zeros = vdupq_n_f32(0);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		uint32x4_t mask = vcgtq_f32(xs, zeros);
		float32x4_t ys = vmulq_n_f32(xs, 0.1);
		float32x4_t zs = vbslq_f32(mask, xs, ys);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) ? X[i] : 0.1 * X[i];
	}
}

void logistic_active_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t ones = vdupq_n_f32(1);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		float32x4_t ys = vaddq_f32(ones, exp_ps(vnegq_f32(xs)));
		float32x4_t zs = vrecpeq_f32(ys);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = 1 / (1 + exp(-X[i]));
	}
}
#endif