#include <stdint.h>
#include <omp.h>
#ifdef __INTEL_SSE__
#include <emmintrin.h>
#include <tmmintrin.h>
#elif __ARM_NEON__
#include <arm_neon.h>
#endif

#include "image.h"
#include "zutils.h"

image *create_image(int width, int height, int nchannels)
{
	image *img = calloc(1, sizeof(image));
	if (!img) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return img;
	}
	
	img->w = width;
	img->h = height;
	img->c = nchannels;
	
	img->data = calloc(width * height * nchannels, sizeof(float));
	if (!img->data) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free_image(img);
	}
	
	return img;
}

void free_image(image *img)
{
	if (!img) return;
	if (img->data) {
		free(img->data);
		img->data = NULL;
	}
	
	free(img);
	img = NULL;
}

#ifdef __INTEL_SSE__
void split_channel_sse(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int pixels_per_load = 16;
	int excess = w - w % pixels_per_load;
	#pragma omp parallel for
	for (int y = 0; y < h; ++y) {
		unsigned char *psrc = src + y * src_pitch;
		unsigned char *pred = dst + y * w;
		unsigned char *pgrn = dst + w * (h + y);
		unsigned char *pblu = dst + w * ((h << 1) + y);
		for (int x = 0; x < excess; x += pixels_per_load) {
			__m128i BGR1 = _mm_loadu_si128((__m128i *)(psrc));
			__m128i BGR2 = _mm_loadu_si128((__m128i *)(psrc + 16));
			__m128i BGR3 = _mm_loadu_si128((__m128i *)(psrc + 32));
			
			__m128i B = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			B = _mm_or_si128(B, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1)));
			B = _mm_or_si128(B, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13)));
			
			__m128i G = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			G = _mm_or_si128(G, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1)));
			G = _mm_or_si128(G, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14)));
			
			__m128i R = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			R = _mm_or_si128(R, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1)));
			R = _mm_or_si128(R, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15)));
			
			_mm_storeu_si128((__m128i *)pred, R);
			_mm_storeu_si128((__m128i *)pgrn, G);
			_mm_storeu_si128((__m128i *)pblu, B);
			
			psrc += 48;
			pred += 16;
			pgrn += 16;
			pblu += 16;
		}
	}
	
	if (excess == w) return;
	
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		for (int y = 0; y < h; ++y) {
			for (int x = excess; x < w; ++x) {
				at[y * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}
#endif

#ifdef __ARM_NEON__
void split_channel_neon(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int pixels_per_load = 16;
	int excess = w - w % pixels_per_load;
	// #pragma omp parallel for
	for (int y = 0; y < h; ++y) {
		unsigned char *psrc = src + y * src_pitch;
		unsigned char *pred = dst + y * w;
		unsigned char *pgrn = dst + w * (h + y);
		unsigned char *pblu = dst + w * ((h << 1) + y);
		for (int x = 0; x < excess; x += pixels_per_load) {
			uint8x16x3_t BGR16 = vld3q_u8(psrc);
			vst1q_u8(pred, BGR16.val[2]);
			vst1q_u8(pgrn, BGR16.val[1]);
			vst1q_u8(pblu, BGR16.val[0]);
			psrc += 48;
			pred += 16;
			pgrn += 16;
			pblu += 16;
		}
	}
	
	if (excess == w) return;
	
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		for (int y = 0; y < h; ++y) {
			for (int x = excess; x < w; ++x) {
				at[y * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}
#endif

void split_channel0(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				at[(h - 1 - y) * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}

void split_channel(const unsigned char *const src, int src_pitch, image *dst)
{
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < dst->c; ++c) {
		float *at = dst->data + swap[c] * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				at[(dst->h - 1 - y) * dst->w + x] = src[y * src_pitch + dst->c * x + c];
			}
		}
	}
}

static inline float32x4_t interpolate(float32x4_t DX, float32x4_t DY, float32x4_t V1,
                               float32x4_t V2, float32x4_t V3, float32x4_t V4)
{
	float32x4_t I1 = vaddq_f32(vmulq_f32(DX, V2), vmulq_f32(vsubq_f32(vdupq_n_f32(1), DX), V1));
	float32x4_t I2 = vaddq_f32(vmulq_f32(DX, V4), vmulq_f32(vsubq_f32(vdupq_n_f32(1), DX), V3));
	return vaddq_f32(vmulq_f32(DY, I2), vmulq_f32(vsubq_f32(vdupq_n_f32(1), DY), I1));
}

void resize_image_neon(uint8_t *src, float *dst, int sw, int sh, int dw, int dh)
{
	float32x4_t scale = vdupq_n_f32((float)sw / dw);
	float32x4_t xbase = {0, 1, 2, 3};
	for (int y = 1; y < dh-1; ++y) {
		for (int x = 0; x < dw; x += 4) {
			float32x4_t X4 = vaddq_f32(vdupq_n_f32((float32_t)x), xbase);
			X4 = vmulq_f32(X4, scale);
			int32x4_t X4_U32 = vcvtq_s32_f32(X4);
			X4 = vsubq_f32(X4, vcvtq_f32_s32(X4_U32));
			
			float32x4_t Y4 = vdupq_n_f32((float32_t)y);
			Y4 = vmulq_f32(Y4, scale);
			int32x4_t Y4_U32 = vcvtq_s32_f32(Y4);
			Y4 = vsubq_f32(Y4, vcvtq_f32_s32(Y4_U32));
			
			float32x4_t V1 = {src[Y4_U32[0] * sw + X4_U32[0]],
			                  src[Y4_U32[1] * sw + X4_U32[1]],
			                  src[Y4_U32[2] * sw + X4_U32[2]],
			                  src[Y4_U32[3] * sw + X4_U32[3]]};

			float32x4_t V2 = {src[Y4_U32[0] * sw + X4_U32[0] + 1],
			                  src[Y4_U32[1] * sw + X4_U32[1] + 1],
							  src[Y4_U32[2] * sw + X4_U32[2] + 1],
			                  src[Y4_U32[3] * sw + X4_U32[3] + 1]};

			float32x4_t V3 = {src[(Y4_U32[0] + 1) * sw + X4_U32[0]],
			                  src[(Y4_U32[1] + 1) * sw + X4_U32[1]],
							  src[(Y4_U32[2] + 1) * sw + X4_U32[2]],
			                  src[(Y4_U32[3] + 1) * sw + X4_U32[3]]};

			float32x4_t V4 = {src[(Y4_U32[0] + 1) * sw + X4_U32[0] + 1],
			                  src[(Y4_U32[1] + 1) * sw + X4_U32[1] + 1],
							  src[(Y4_U32[2] + 1) * sw + X4_U32[2] + 1],
							  src[(Y4_U32[3] + 1) * sw + X4_U32[3] + 1]};

			float32x4_t J4 = interpolate(X4, Y4, V1, V2, V3, V4);
			
			vst1q_f32(dst + y * dw + x, J4);
		}
	}
}

void resize_image0(unsigned char *src, image *dst, int sw, int sh)
{
	float s = (float)sw / dst->w;
	for (int c = 0; c < dst->c; ++c) {
		unsigned char *src_at = src + c * sw * sh;
		float *dst_at = dst->data + c * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				float sx = s * x;
				float sy = s * y;
				int left = (int)sx;
				int top = (int)sy;
				float i1 = (sx - left) * src_at[top * sw + left + 1] +
				       (left + 1 - sx) * src_at[top * sw + left];
				float i2 = (sx - left) * src_at[(top + 1) * sw + left + 1] +
					   (left + 1 - sx) * src_at[(top + 1) * sw + left];
				dst_at[y * dst->w + x] = (sy - top) * i2 + (top + 1 - sy) * i1;
			}
		}
	}
}

void resize_image(image *src, image *dst)
{
	float s = (float)src->w / dst->w;
	for (int c = 0; c < dst->c; ++c) {
		float *src_at = src->data + c * src->w * src->h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				float sx = s * x;
				float sy = s * y;
				int left = (int)sx;
				int top = (int)sy;
				float i1 = (sx - left) * src_at[top * src->w + left + 1] +
				       (left + 1 - sx) * src_at[top * src->w + left];
				float i2 = (sx - left) * src_at[(top + 1) * src->w + left + 1] +
					   (left + 1 - sx) * src_at[(top + 1) * src->w + left];
				dst_at[y * dst->w + x] = (sy - top) * i2 + (top + 1 - sy) * i1;
			}
		}
	}
}

void embed_image(image *src, image *dst)
{
	int dx = (dst->w - src->w) / 2;
	int dy = (dst->h - src->h) / 2;
	for (int c = 0; c < src->c; ++c) {
		float *src_at = src->data + c * src->w * src->h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		for (int y = 0; y < src->h; ++y) {
			for (int x = 0; x < src->w; ++x) {
				dst_at[(y + dy) * dst->w + x + dx] = src_at[y * src->w + x] / 255;
			}
		}
	}
}

void set_image(image *img, float val)
{
	size_t size = img->w * img->h * img->c * sizeof(float);
	mset((char *const)img->data, size, (const char *const)&val, sizeof(float));
}

void vertical_mirror(image *img)
{
	int hh = img->h >> 1;
	for (int c = 0; c < img->c; ++c) {
		float *at = img->data + c * img->w * img->h;
		for (int y = 0; y < hh; ++y) {
			for (int x = 0; x < img->w; ++x) {
				float swap = at[y * img->w + x];
				at[y * img->w + x] = at[(img->h - 1 - y) * img->w + x];
				at[(img->h - 1 - y) * img->w + x] = swap;
			}
		}
	}
}

void swap_channel(image *img)
{
	int offset = img->w * img->h * 2;
	for (int y = 0; y < img->h; ++y) {
		for (int x = 0; x < img->w; ++x) {
			float swap = img->data[y * img->w + x];
			img->data[y * img->w + x] = img->data[y * img->w + x + offset];
			img->data[y * img->w + x + offset] = swap;
		}
	}
}