#ifndef _BITMAP_H_
#define _BITMAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

struct __bitmap;
typedef struct __bitmap bitmap;

bitmap *read_bmp(const char *filename);
bitmap *create_bmp(const char *const data, int width, int height, int bits_per_pixel);
unsigned char *get_bmp_data(bitmap *bmp);
int get_bmp_width(bitmap *bmp);
int get_bmp_height(bitmap *bmp);
int get_bmp_bit_count(bitmap *bmp);
int get_bmp_pitch(bitmap *bmp);
void save_bmp(bitmap *bmp, const char *filename);
void delete_bmp(bitmap *bmp);

#ifdef __cplusplus
}
#endif

#endif