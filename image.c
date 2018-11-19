#include "image.h"

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

void split(const unsigned char *const src, float *const dst, int width, int height, int nchannels)
{
	for (int c = 0; c < nchannels; ++c) {
		float *at = dst + c * width * height;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				at[y * width + x] = src[nchannels * (y * width + x) + c];
			}
		}
	}
}

void resize_image(image src, image *dst)
{
	float s = src.w / dst->w;
	for (int c = 0; c < dst->c; ++c) {
		float *src_at = src.data + c * src.w * src.h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		for (int y = 0; y < dst->h; ++y) {
			for (int x = 0; x < dst->w; ++x) {
				float sx = s * x;
				float sy = s * y;
				int left = (int)sx;
				int top = (int)sy;
				float i1 = (sx - left) * src_at[top * src.w + left + 1] +
					(left + 1 - sx) * src_at[top * src.w + left];
				float i2 = (sx - left) * src_at[(top + 1) * src.w + left + 1] +
					(left + 1 - sx) * src_at[(top + 1) * src.w + left];
				dst_at[y * dst->w + x] = (sy - top) * i2 + (top + 1 - sy) * i1;
			}
		}
	}
}