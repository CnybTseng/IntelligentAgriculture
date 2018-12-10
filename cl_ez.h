/** @file cl_ez.h
 ** @brief OpenCL easy using method.
 ** @author Zhiwei Zeng
 ** @date 2018.10.29
 **/

/*
Copyright (C) 2018 Zhiwei Zeng.
Copyright (C) 2018 Chengdu ZLT Technology Co., Ltd.
All rights reserved.

This file is part of the xxx toolkit and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __OPENCL_CL_EZ_H
#define __OPENCL_CL_EZ_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "opencl.h"

/** @typedef struct cl_platform_layer
 ** @brief OpenCL platform layer.
 **/
typedef struct {
	cl_platform_id *platforms;		/*< a list of OpenCL platforms. */
	cl_uint nplatforms;				/*< the number of OpenCL platforms. */
	cl_device_id *devices;			/*< a list of OpenCL devices on the first platform. */
	cl_uint ndevices;				/*< the number of OpenCL devices on the first platform. */
	cl_context context;				/*< OpenCL context of the first device of the first platform. */
} cl_platform_layer;

/** @typedef struct cl_runtime.
 ** @brief OpenCL runtime.
 **/
typedef struct {
	cl_command_queue cmdqueue;		/*< OpenCL command queue. */
	cl_program program;				/*< OpenCL program object for a context. */
	cl_kernel kernel;				/*< OpenCL kernel object. */
} cl_runtime;

/** @name OpenCL platform layer and runtime creation, query, and destroying.
 ** @{ */
cl_platform_layer *cl_create_platform_layer();
void cl_destroy_platform_layer(cl_platform_layer *platform_layer);
void cl_get_platform_layer_info(cl_platform_layer *platform_layer);
cl_runtime *cl_create_runtime(cl_device_id device, cl_context context, const char *filename);
void cl_destroy_runtime(cl_runtime *runtime);
/** @} */

#ifdef __cplusplus
}
#endif

#endif