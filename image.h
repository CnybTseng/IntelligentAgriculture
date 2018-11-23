#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

image *create_image(int width, int height, int nchannels);
void free_image(image *img);
void split_channel(const unsigned char *const src, int src_pitch, image *dst);
void resize_image(image *src, image *dst);
void embed_image(image *src, image *dst);
void set_image(image *img, float val);
void vertical_mirror(image *img);
void swap_channel(image *img);

#ifdef __cplusplus
}
#endif

#endif