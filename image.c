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
	#pragma omp parallel for
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

void split_channel(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		#pragma omp parallel for
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				at[(h - 1 - y) * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}

#ifdef __ARM_NEON__
static inline uint8x8_t interpolate(uint16x8_t Dx, uint16x8_t Dy, uint8x8_t V1,
                                    uint8x8_t V2, uint8x8_t V3, uint8x8_t V4)
{
	uint16x4_t Dx_u16_l4 = vget_low_u16(Dx);
	uint16x4_t Dx_u16_h4 = vget_high_u16(Dx);
	uint32x4_t Dx_u32_l4 = vmovl_u16(Dx_u16_l4);
	uint32x4_t Dx_u32_h4 = vmovl_u16(Dx_u16_h4);
		
	uint16x4_t Dy_u16_l4 = vget_low_u16(Dy);
	uint16x4_t Dy_u16_h4 = vget_high_u16(Dy);
	uint32x4_t Dy_u32_l4 = vmovl_u16(Dy_u16_l4);
	uint32x4_t Dy_u32_h4 = vmovl_u16(Dy_u16_h4);
	
	uint32x4_t _Dx_u32_l4 = vsubq_u32(vdupq_n_u32(4096), Dx_u32_l4);
	uint32x4_t _Dx_u32_h4 = vsubq_u32(vdupq_n_u32(4096), Dx_u32_h4);
	uint32x4_t _Dy_u32_l4 = vsubq_u32(vdupq_n_u32(4096), Dy_u32_l4);
	uint32x4_t _Dy_u32_h4 = vsubq_u32(vdupq_n_u32(4096), Dy_u32_h4);
	
	uint16x8_t V1_u16 = vmovl_u8(V1);
	uint16x8_t V2_u16 = vmovl_u8(V2);
	uint16x8_t V3_u16 = vmovl_u8(V3);
	uint16x8_t V4_u16 = vmovl_u8(V4);
	
	uint16x4_t V1_u16_l4 = vget_low_u16(V1_u16);
	uint16x4_t V1_u16_h4 = vget_high_u16(V1_u16);
	uint32x4_t V1_u32_l4 = vmovl_u16(V1_u16_l4);
	uint32x4_t V1_u32_h4 = vmovl_u16(V1_u16_h4);
	
	uint16x4_t V2_u16_l4 = vget_low_u16(V2_u16);
	uint16x4_t V2_u16_h4 = vget_high_u16(V2_u16);
	uint32x4_t V2_u32_l4 = vmovl_u16(V2_u16_l4);
	uint32x4_t V2_u32_h4 = vmovl_u16(V2_u16_h4);
	
	uint16x4_t V3_u16_l4 = vget_low_u16(V3_u16);
	uint16x4_t V3_u16_h4 = vget_high_u16(V3_u16);
	uint32x4_t V3_u32_l4 = vmovl_u16(V3_u16_l4);
	uint32x4_t V3_u32_h4 = vmovl_u16(V3_u16_h4);
	
	uint16x4_t V4_u16_l4 = vget_low_u16(V4_u16);
	uint16x4_t V4_u16_h4 = vget_high_u16(V4_u16);
	uint32x4_t V4_u32_l4 = vmovl_u16(V4_u16_l4);
	uint32x4_t V4_u32_h4 = vmovl_u16(V4_u16_h4);
	
	uint32x4_t I1_u32_l4 = vaddq_u32(vmulq_u32(Dx_u32_l4, V2_u32_l4), vmulq_u32(_Dx_u32_l4, V1_u32_l4));
	uint32x4_t I1_u32_h4 = vaddq_u32(vmulq_u32(Dx_u32_h4, V2_u32_h4), vmulq_u32(_Dx_u32_h4, V1_u32_h4));
	
	uint32x4_t I2_u32_l4 = vaddq_u32(vmulq_u32(Dx_u32_l4, V4_u32_l4), vmulq_u32(_Dx_u32_l4, V3_u32_l4));
	uint32x4_t I2_u32_h4 = vaddq_u32(vmulq_u32(Dx_u32_h4, V4_u32_h4), vmulq_u32(_Dx_u32_h4, V3_u32_h4));
	
	uint32x4_t II_u32_l4 = vaddq_u32(vmulq_u32(Dy_u32_l4, I2_u32_l4), vmulq_u32(_Dy_u32_l4, I1_u32_l4));
	uint32x4_t II_u32_h4 = vaddq_u32(vmulq_u32(Dy_u32_h4, I2_u32_h4), vmulq_u32(_Dy_u32_h4, I1_u32_h4));
	
	uint16x4_t II_u16_l4 = vshrn_n_u32(II_u32_l4, 16);
	uint16x4_t II_u16_h4 = vshrn_n_u32(II_u32_h4, 16);
	
	II_u16_l4 = vrshr_n_u16(II_u16_l4, 8);
	II_u16_h4 = vrshr_n_u16(II_u16_h4, 8);
	
	uint16x8_t II_u16 = vcombine_u16(II_u16_l4, II_u16_h4);
	
	return vqmovn_u16(II_u16);
}
#endif

#ifdef __ARM_NEON__
static inline uint8x8_t batch_read_pixel(unsigned char *ptr, int pitch, int16x8_t x, short y)
{
	uint8x8_t Pix = {
		ptr[y * pitch + x[0]],
		ptr[y * pitch + x[1]],
		ptr[y * pitch + x[2]],
		ptr[y * pitch + x[3]],
		ptr[y * pitch + x[4]],
		ptr[y * pitch + x[5]],
		ptr[y * pitch + x[6]],
		ptr[y * pitch + x[7]]
	};
	
	return Pix;
}

#endif

#ifdef __ARM_NEON__
void resize_image_neon(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                       int dst_w, int dst_h, int nchannels)
{
	float s = (float)src_w / dst_w;
	float32x4_t delta_l4 = {0, 1, 2, 3};
	float32x4_t delta_h4 = {4, 5, 6, 7};
	int16x8_t minx = vdupq_n_s16(0);
	int16x8_t maxx = vdupq_n_s16(src_w - 2);
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = s * (y + 0.5) - 0.5;
			short top = (short)sy;
			uint16x8_t Dy = vdupq_n_u16((unsigned short)((sy -top) * 4096));
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; x += 8) {
				float32x4_t X_f32_l4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_l4);
				float32x4_t X_f32_h4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_h4);
				
				X_f32_l4 = vsubq_f32(vmulq_n_f32(X_f32_l4, s), vdupq_n_f32(0.5));
				X_f32_h4 = vsubq_f32(vmulq_n_f32(X_f32_h4, s), vdupq_n_f32(0.5));
				
				int32x4_t X_s32_l4 = vcvtq_s32_f32(X_f32_l4);
				int32x4_t X_s32_h4 = vcvtq_s32_f32(X_f32_h4);
				
				int16x4_t X_s16_l4 = vmovn_s32(X_s32_l4);
				int16x4_t X_s16_h4 = vmovn_s32(X_s32_h4);
				
				float32x4_t Dx_f32_l4 = vsubq_f32(X_f32_l4, vcvtq_f32_s32(X_s32_l4));
				float32x4_t Dx_f32_h4 = vsubq_f32(X_f32_h4, vcvtq_f32_s32(X_s32_h4));
				
				Dx_f32_l4 = vmulq_n_f32(Dx_f32_l4, 4096);
				Dx_f32_h4 = vmulq_n_f32(Dx_f32_h4, 4096);
				
				uint32x4_t Dx_u32_l4 = vcvtq_u32_f32(Dx_f32_l4);
				uint32x4_t Dx_u32_h4 = vcvtq_u32_f32(Dx_f32_h4);
				
				uint16x4_t Dx_u16_l4 = vmovn_u32(Dx_u32_l4);
				uint16x4_t Dx_u16_h4 = vmovn_u32(Dx_u32_h4);

				uint16x8_t Dx = vcombine_u16(Dx_u16_l4, Dx_u16_h4);
				
				int16x8_t left = vcombine_s16(X_s16_l4, X_s16_h4);
				left = vminq_s16(vmaxq_s16(left, minx), maxx);
				
				uint8x8_t V1 = batch_read_pixel(src_at, src_w, left, top);
				uint8x8_t V2 = batch_read_pixel(src_at, src_w, vaddq_s16(left, vdupq_n_s16(1)), top);
				uint8x8_t V3 = batch_read_pixel(src_at, src_w, left, top + 1);
				uint8x8_t V4 = batch_read_pixel(src_at, src_w, vaddq_s16(left, vdupq_n_s16(1)), top + 1);

				uint8x8_t J8 = interpolate(Dx, Dy, V1, V2, V3, V4);
				
				vst1_u8(dst_at + y * dst_w + x, J8);
			}
		}
	}
}
#endif

#ifdef __ARM_NEON__
void make_bilinear_interp_table(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                                short *y_tab, unsigned short *dx_tab, unsigned short *dy_tab)
{
	float s = (float)src_w / dst_w;
	float32x4_t delta_l4 = {0, 1, 2, 3};
	float32x4_t delta_h4 = {4, 5, 6, 7};
	int16x8_t minx = vdupq_n_s16(0);
	int16x8_t maxx = vdupq_n_s16(src_w - 2);
	
	#pragma omp parallel for
	for (int y = 0; y < dst_h; ++y) {
		float sy = s * (y + 0.5) - 0.5;
		short top = (short)sy;
		uint16x8_t Dy = vdupq_n_u16((unsigned short)((sy -top) * 4096));
		if (top < 0) top = 0;
		if (top > src_h - 2) top = src_h - 2;
		for (int x = 0; x < dst_w; x += 8) {
			float32x4_t X_f32_l4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_l4);
			float32x4_t X_f32_h4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_h4);
			
			X_f32_l4 = vsubq_f32(vmulq_n_f32(X_f32_l4, s), vdupq_n_f32(0.5));
			X_f32_h4 = vsubq_f32(vmulq_n_f32(X_f32_h4, s), vdupq_n_f32(0.5));
			
			int32x4_t X_s32_l4 = vcvtq_s32_f32(X_f32_l4);
			int32x4_t X_s32_h4 = vcvtq_s32_f32(X_f32_h4);
			
			int16x4_t X_s16_l4 = vmovn_s32(X_s32_l4);
			int16x4_t X_s16_h4 = vmovn_s32(X_s32_h4);
			
			int16x8_t left = vcombine_s16(X_s16_l4, X_s16_h4);
			left = vminq_s16(vmaxq_s16(left, minx), maxx);
			
			float32x4_t Dx_f32_l4 = vsubq_f32(X_f32_l4, vcvtq_f32_s32(X_s32_l4));
			float32x4_t Dx_f32_h4 = vsubq_f32(X_f32_h4, vcvtq_f32_s32(X_s32_h4));
			
			Dx_f32_l4 = vmulq_n_f32(Dx_f32_l4, 4096);
			Dx_f32_h4 = vmulq_n_f32(Dx_f32_h4, 4096);
			
			uint32x4_t Dx_u32_l4 = vcvtq_u32_f32(Dx_f32_l4);
			uint32x4_t Dx_u32_h4 = vcvtq_u32_f32(Dx_f32_h4);
			
			uint16x4_t Dx_u16_l4 = vmovn_u32(Dx_u32_l4);
			uint16x4_t Dx_u16_h4 = vmovn_u32(Dx_u32_h4);

			uint16x8_t Dx = vcombine_u16(Dx_u16_l4, Dx_u16_h4);
			
			vst1q_s16(x_tab, left);
			vst1q_s16(y_tab, vdupq_n_s16(top));
			vst1q_u16(dx_tab, Dx);
			vst1q_u16(dy_tab, Dy);
			
			x_tab += 8;
			y_tab += 8;
			dx_tab += 8;
			dy_tab += 8;
		}
	}
}

void package_neighbor_pixle(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                            short *y_tab, unsigned char *src, unsigned char *pack)
{
	for (int c = 0; c < 3; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *pack_at = pack + c * dst_w * dst_h * 4;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			for (int x = 0; x < dst_w; ++x) {
				int id = y * dst_w + x;
				pack_at[y * dst_w * 4 + 4 * x]     = src_at[y_tab[id] * src_w + x_tab[id]];
				pack_at[y * dst_w * 4 + 4 * x + 1] = src_at[y_tab[id] * src_w + x_tab[id] + 1];
				pack_at[y * dst_w * 4 + 4 * x + 2] = src_at[(y_tab[id] + 1) * src_w + x_tab[id]];
				pack_at[y * dst_w * 4 + 4 * x + 3] = src_at[(y_tab[id] + 1) * src_w + x_tab[id] + 1];
			}
		}
	}
}

void resize_image_neon_faster(unsigned char *pack, unsigned char *dst, int dst_w, int dst_h,
                              int nchannels, unsigned short *dx_tab, unsigned short *dy_tab)
{
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *pack_at = pack + c * dst_w * dst_h * 4;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		unsigned short *pdx = dx_tab;
		unsigned short *pdy = dy_tab;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			for (int x = 0; x < dst_w; x += 8) {
				uint8x8x4_t V = vld4_u8(pack_at);
				uint16x8_t Dx = vld1q_u16(pdx);
				uint16x8_t Dy = vld1q_u16(pdy);
				uint8x8_t IV = interpolate(Dx, Dy, V.val[0], V.val[1], V.val[2], V.val[3]);
				vst1_u8(dst_at + y * dst_w + x, IV);
				pack_at += 32;
				pdx += 8;
				pdy += 8;
			}
		}
	}
}
#endif

void resize_image_hv(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                     int dst_w, int dst_h, int nchannels)
{
	float sh = (float)(src_w - 1) / (dst_w - 1);
	float sv = (float)(src_h - 1) / (dst_h - 1);
	unsigned char *resized = calloc(dst_w * src_h * nchannels, sizeof(unsigned char));
	if (!resized) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *rsz_at = resized + c * dst_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		
		// 水平缩放
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			for (int x = 0; x < dst_w; ++x) {
				float sx = sh * (x + 0.5) - 0.5;
				int left = (int)sx;
				int dx = (int)((sx - left) * 4096);
				if (left < 0) left = 0;
				if (left > src_w - 2) left = src_w - 2;
				int val = dx * src_at[y * src_w + left + 1] + (4096 - dx) * src_at[y * src_w + left];
				rsz_at[y * dst_w + x] = val >> 12;
			}
		}
		
		// 垂直缩放
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = sv * (y + 0.5) - 0.5;
			int top = (int)sy;
			int dy = (int)((sy - top) * 4096);
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; ++x) {
				int val = dy * rsz_at[(top + 1) * dst_w + x] + (4096 - dy) * rsz_at[top * dst_w + x];
				dst_at[y * dst_w + x] = val >> 12;
			}
		}
	}
	
	free(resized);
}

void resize_image(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                  int dst_w, int dst_h, int nchannels)
{
	float s = (float)src_w / dst_w;
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = s * (y + 0.5) - 0.5;
			int top = (int)sy;
			int dy = (int)((sy -top) * 4096);
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; ++x) {
				float sx = s * (x + 0.5) - 0.5;
				int left = (int)sx;
				int dx = (int)((sx - left) * 4096);
				if (left < 0) left = 0;
				if (left > src_w - 2) left = src_w - 2;
				int v1 = dx * src_at[top * src_w + left + 1] + (4096 - dx) * src_at[top * src_w + left];
				int v2 = dx * src_at[(top + 1) * src_w + left + 1] + (4096 - dx) * src_at[(top + 1) * src_w + left];
				dst_at[y * dst_w + x] = (dy * v2 + (4096 - dy) * v1) >> 24;
			}
		}
	}
}

#ifdef __ARM_NEON__
void embed_image_neon(unsigned char *src, image *dst, int src_w, int src_h)
{
	float norm = 1.0 / 255;
	int dx = (dst->w - src_w) / 2;
	int dy = (dst->h - src_h) / 2;
	int pixels_per_load = 16;
	int excess = src_w - src_w % pixels_per_load;
	
	for (int c = 0; c < dst->c; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			unsigned char *psrc = src_at + y * src_w;
			float *pdst = dst_at + (y + dy) * dst->w + dx;
			for (int x = 0; x < excess; x += pixels_per_load) {
				uint8x16_t V_u8 = vld1q_u8(psrc);
				
				uint8x8_t V_u8_l8 = vget_low_u8(V_u8);
				uint8x8_t V_u8_h8 = vget_high_u8(V_u8);
				
				uint16x8_t V_u16_l8 = vmovl_u8(V_u8_l8);
				uint16x8_t V_u16_h8 = vmovl_u8(V_u8_h8);
				
				uint16x4_t V_u16_ll4 = vget_low_u16(V_u16_l8);
				uint16x4_t V_u16_lh4 = vget_high_u16(V_u16_l8);
				
				uint16x4_t V_u16_hl4 = vget_low_u16(V_u16_h8);
				uint16x4_t V_u16_hh4 = vget_high_u16(V_u16_h8);
				
				uint32x4_t V_u32_ll4 = vmovl_u16(V_u16_ll4);
				uint32x4_t V_u32_lh4 = vmovl_u16(V_u16_lh4);
				
				uint32x4_t V_u32_hl4 = vmovl_u16(V_u16_hl4);
				uint32x4_t V_u32_hh4 = vmovl_u16(V_u16_hh4);
				
				float32x4_t V_f32_ll4 = vcvtq_f32_u32(V_u32_ll4);
				float32x4_t V_f32_lh4 = vcvtq_f32_u32(V_u32_lh4);
				
				float32x4_t V_f32_hl4 = vcvtq_f32_u32(V_u32_hl4);
				float32x4_t V_f32_hh4 = vcvtq_f32_u32(V_u32_hh4);
				
				V_f32_ll4 = vmulq_n_f32(V_f32_ll4, norm);
				V_f32_lh4 = vmulq_n_f32(V_f32_lh4, norm);
				
				V_f32_hl4 = vmulq_n_f32(V_f32_hl4, norm);
				V_f32_hh4 = vmulq_n_f32(V_f32_hh4, norm);
				
				vst1q_f32(pdst,      V_f32_ll4);
				vst1q_f32(pdst +  4, V_f32_lh4);
				vst1q_f32(pdst +  8, V_f32_hl4);
				vst1q_f32(pdst + 12, V_f32_hh4);
				
				psrc += pixels_per_load;
				pdst += pixels_per_load;
			}
		}
		
		if (excess == src_w) continue;
		
		for (int y = 0; y < src_h; ++y) {
			for (int x = excess; x < src_w; ++x) {
				dst_at[(y + dy) * dst->w + x + dx] = src_at[y * src_w + x] * norm;
			}
		}
	}
}
#endif

void embed_image(unsigned char *src, image *dst, int src_w, int src_h)
{
	int dx = (dst->w - src_w) / 2;
	int dy = (dst->h - src_h) / 2;
	for (int c = 0; c < dst->c; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			for (int x = 0; x < src_w; ++x) {
				dst_at[(y + dy) * dst->w + x + dx] = src_at[y * src_w + x] / 255.0f;
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