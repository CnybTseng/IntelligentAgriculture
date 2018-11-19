#ifndef _BMP_H_
#define _BMP_H_

#ifdef __cplusplus
extern "C"
{
#endif

/** @typedef struct BMP.
 ** @brief bit map picture structure.
 **/
typedef struct
{
	unsigned char *data;		/**< image data. */
	unsigned int   width;		/**< image width. */
	unsigned int   height;		/**< image height. */
	unsigned int   bit_count;	/**< bits per pixel. */
}BMP;

/** @name Read, write, and delete.
 ** @{ */
BMP *bmp_read(const char *path);
BMP *bmp_create(const char *data,
                unsigned int width,
				unsigned int height,
				unsigned int bit_count);
void bmp_set(const char *data,
             unsigned int len,
			 BMP *bmp);
int  bmp_write(const BMP *bmp, const char *path);
void bmp_delete(BMP *bmp);
/** @} */

#ifdef __cplusplus
}
#endif

#endif