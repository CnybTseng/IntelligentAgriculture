#ifndef _NEON_MATH_H_
#define _NEON_MATH_H_

#include <arm_neon.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef float32x4_t v4sf;

v4sf log_ps(v4sf x);
v4sf exp_ps(v4sf x);
v4sf sin_ps(v4sf x);
v4sf cos_ps(v4sf x);

#ifdef __cplusplus
}
#endif

#endif