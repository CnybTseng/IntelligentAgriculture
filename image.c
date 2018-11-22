#include <emmintrin.h>
#include <tmmintrin.h>
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

void split_channel_sse2(const unsigned char *const src, unsigned char *dst, int w, int h)
{
	int ppl = 48;
	unsigned char *pr = dst;
	unsigned char *pg = dst + w * h;
	unsigned char *pb = dst + 2 * w * h;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; x += ppl) {
			__m128i RGB1 = _mm_loadu_si128((__m128i *)src + y * w);
			__m128i RGB2 = _mm_loadu_si128((__m128i *)src + y * w + 16);
			__m128i RGB3 = _mm_loadu_si128((__m128i *)src + y * w + 32);
			
			__m128i R = _mm_shuffle_epi8(RGB1, _mm_setr_epi8(
				0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			R = _mm_or_si128(R, _mm_shuffle_epi8(RGB2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1)));
			R = _mm_or_si128(R, _mm_shuffle_epi8(RGB3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13)));
			
			__m128i G = _mm_shuffle_epi8(RGB1, _mm_setr_epi8(
				1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			G = _mm_or_si128(G, _mm_shuffle_epi8(RGB2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1)));
			G = _mm_or_si128(G, _mm_shuffle_epi8(RGB3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14)));
			
			__m128i B = _mm_shuffle_epi8(RGB1, _mm_setr_epi8(
				2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			B = _mm_or_si128(B, _mm_shuffle_epi8(RGB2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1)));
			B = _mm_or_si128(B, _mm_shuffle_epi8(RGB3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15)));
			
			_mm_storeu_si128((__m128i *)pr, R);
			_mm_storeu_si128((__m128i *)pg, G);
			_mm_storeu_si128((__m128i *)pb, B);
		}
	}
}

void split_channel(const unsigned char *const src, image *dst)
{
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < dst->c; ++c) {
		float *at = dst->data + swap[c] * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				at[(dst->h - 1 - y) * dst->w + x] = src[dst->c * (y * dst->w + x) + c];
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