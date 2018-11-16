#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void activate(float *X, int n, ACTIVATION activation);

#ifdef __cplusplus
}
#endif

#endif