#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

image *create_image(int width, int height, int nchannels);
void free_image(image *img);
void split_channel(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
void resize_image(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                  int dst_w, int dst_h, int nchannels);
void embed_image(unsigned char *src, image *dst, int src_w, int src_h);
void set_image(image *img, float val);
void vertical_mirror(image *img);
void swap_channel(image *img);

#ifdef __cplusplus
}
#endif

#endif