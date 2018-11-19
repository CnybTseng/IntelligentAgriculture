#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "convnet.h"
#include "bmp.h"

#ifdef __cplusplus
extern "C"
{
#endif

image *create_image(int width, int height, int nchannels);
void free_image(image *img);
void split(const unsigned char *const src, float *const dst, int width, int height, int nchannels);
void resize_image(image src, image *dst);

#ifdef __cplusplus
}
#endif

#endif