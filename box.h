#ifndef _BOX_H_
#define _BOX_H_

#include "znet.h"

#ifdef __cplusplus
extern "C"
{
#endif

int equ_val(void *v1, void *v2);
void free_val(void *v);
float box_intersection(box *b1, box *b2);
float box_union(box *b1, box *b2);
float IOU(box *b1, box *b2);
float penalize_score(float sigma, float score, float iou);

#ifdef __cplusplus
}
#endif

#endif