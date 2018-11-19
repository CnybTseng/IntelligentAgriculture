#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>

#include "bmp.h"

#define BM 0x4D42

#pragma pack(1)

typedef struct
{
	unsigned short ftype;
	unsigned int   fsize;
	unsigned short freserved1;
	unsigned short freserved2;
	unsigned int   foff_bits;
}BitMapFileHeader;

typedef struct
{
	unsigned int   isize;
	unsigned long  iwidth;
	unsigned long  iheight;
	unsigned short iplanes;
	unsigned short ibit_count;
	unsigned int   icompression;
	unsigned int   isize_image;
	unsigned int   ix_pels_per_meter;
	unsigned int   iy_pels_per_meter;
	unsigned int   iclr_used;
	unsigned int   iclr_important;
}BitMapInfoHeader;

typedef struct
{
	unsigned char rgb_blue;
	unsigned char rgb_green;
	unsigned char rgb_red;
	unsigned char rgb_reserved;
}RgbQuad;

#pragma pack()

BMP *bmp_read(const char *path)
{
	BMP *bmp = NULL;
	struct stat fs;
	FILE *fp = NULL;
	BitMapFileHeader bmfh;
	BitMapInfoHeader bmih;
	unsigned int ncolors;
	unsigned int bpp;
	
	if (stat(path, &fs)) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	fp = fopen(path, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	/* read bit map file header. */
	fread(&bmfh, sizeof(BitMapFileHeader), 1, fp);
	if (bmfh.ftype != BM) {
		fprintf(stderr, "Type unknown[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		return bmp;
	}
	
	/* read bit map info header. */
	fread(&bmih, sizeof(BitMapInfoHeader), 1, fp);
	
	bmp = (BMP *)malloc(sizeof(BMP));
	if (!bmp) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		return bmp;
	}
	
	bmp->width = (unsigned int)bmih.iwidth;
	bmp->height = (unsigned int)bmih.iheight;
	bmp->bit_count = bmih.ibit_count;

	ncolors = 1 << bmih.ibit_count;
	bpp = bmih.ibit_count >> 3;
	
	bmp->data = (unsigned char *)malloc(bpp * bmp->width * bmp->height);
	if (!bmp->data) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		bmp_delete(bmp);
		return bmp;
	}

	/* skip palette if necessary. */
	if (bmp->bit_count < 24) {
		fseek(fp, ncolors * sizeof(RgbQuad), SEEK_CUR);
	}
	
	/* read image data. */
	fread(bmp->data, bpp , bmp->width * bmp->height, fp);
	fclose(fp);
	
	return bmp;
}

BMP *bmp_create(const char *data,
                unsigned int width,
				unsigned int height,
				unsigned int bit_count)
{
	BMP *bmp = NULL;
	unsigned int bpp;
	
	bpp = bit_count >> 3;

	if (!data) {
		fprintf(stderr, "BMP NULL[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	bmp = (BMP *)malloc(sizeof(BMP));
	if (!bmp) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	bmp->data = (unsigned char *)malloc(bpp * width * height);
	if (!bmp->data) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		free(bmp);
		bmp = NULL;
		return bmp;
	}
	
	memmove(bmp->data, data, bpp * width * height);
	bmp->width = width;
	bmp->height = height;
	bmp->bit_count = bit_count;
	
	return bmp;
}

void bmp_set(const char *data,
             unsigned int len,
			 BMP *bmp)
{
	if (!data || !bmp) {
		return;
	}
	
	memmove(bmp->data, data, len);
}

int bmp_write(const BMP *bmp, const char *path)
{
	FILE *fp = NULL;
	unsigned int ncolors;
	unsigned int bpp;
	BitMapFileHeader bmfh;
	BitMapInfoHeader bmih;
	RgbQuad *rq = NULL;
	unsigned int i = 0;
	
	ncolors = 1 << bmp->bit_count;
	bpp = bmp->bit_count >> 3;
	
	fp = fopen(path, "wb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	/* set bit map file header. */
	bmfh.ftype = BM;
	bmfh.fsize = sizeof(BitMapFileHeader) +\
		sizeof(BitMapInfoHeader) +\
		bpp * bmp->width * bmp->height;
	
	if (bmp->bit_count < 24) {
		bmfh.fsize += ncolors * sizeof(RgbQuad);
	}
	
	bmfh.freserved1 = 0;
	bmfh.freserved2 = 0;
	bmfh.foff_bits  = sizeof(BitMapFileHeader) +\
		sizeof(BitMapInfoHeader);
	
	if (bmp->bit_count < 24) {
		bmfh.foff_bits += ncolors * sizeof(RgbQuad);
	}
	
	/* set bit map info header. */
	bmih.isize = sizeof(BitMapInfoHeader);
	bmih.iwidth = bmp->width;
	bmih.iheight = bmp->height;
	bmih.iplanes = 1;
	bmih.ibit_count = bmp->bit_count;
	bmih.icompression = 0;
	bmih.isize_image = bpp * bmp->width * bmp->height;
	bmih.ix_pels_per_meter = 0;
	bmih.iy_pels_per_meter = 0;
	bmih.iclr_used = 0;
	bmih.iclr_important = 0;
	
	fwrite(&bmfh, sizeof(BitMapFileHeader), 1, fp);
	fwrite(&bmih, sizeof(BitMapInfoHeader), 1, fp);

	if (bmp->bit_count < 24) {
		/* set palette. */
		rq = (RgbQuad *)malloc(ncolors * sizeof(RgbQuad));
		if (!rq) {
			fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
			fclose(fp);
			return -1;
		}
		
		for (i = 0; i < ncolors; i++) {
			memset(&rq[i], i, sizeof(RgbQuad));
		}
		
		fwrite(rq, sizeof(RgbQuad), ncolors, fp);

		free(rq);
		rq = NULL;
	}
	
	/* write image data. */
	fwrite(bmp->data, bpp, bmp->width * bmp->height, fp);
	fclose(fp);
	
	return 0;
}

void bmp_delete(BMP *bmp)
{
	if (bmp) {
		if (bmp->data) {
			free(bmp->data);
			bmp->data = NULL;
		}
		
		free(bmp);
		bmp = NULL;
	}
}