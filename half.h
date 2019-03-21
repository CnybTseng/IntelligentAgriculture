#ifndef _HALF_H_
#define _HALF_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef OPENCL
#include "CL/opencl.h"

#ifdef FLOAT
#	define HOST_TO_DEVICE(val) val
#	define DEVICE_TO_HOST(val) val
#else
#	define HOST_TO_DEVICE(val) to_half(val)
#	define DEVICE_TO_HOST(val) to_float(val)
#endif

cl_half to_half(float f);
float to_float(cl_half h);

#endif

#ifdef __cplusplus
}
#endif

#endif
