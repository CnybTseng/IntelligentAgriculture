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

void split_channel(const unsigned char *const src, image *dst)
{
	for (int c = 0; c < dst->c; ++c) {
		float *at = dst->data + c * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				at[y * dst->w + x] = src[dst->c * (y * dst->w + x) + c];
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
		for (int y = 0; y < src->h; ++y) {
			for (int x = 0; x < src->w; ++x) {
				dst->data[(y + dy) * dst->w + x + dx] = src->data[y * src->w + x] / 255;
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
				at[y * img->w + x] = at[(img->h - y) * img->w + x];
				at[(img->h - y) * img->w + x] = swap;
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