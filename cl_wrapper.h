#ifdef OPENCL
#ifndef _CL_WRAPPER_H_
#define _CL_WRAPPER_H_

#include "CL/opencl.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define CL_WRAPPER_FILE_OPEN_FAIL -100
#define CL_WRAPPER_CALLOC_FAIL    -101

typedef struct {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue command_queue;
} cl_wrapper;

cl_wrapper cl_create_wrapper(cl_int *errcode);
cl_program cl_make_wrapper_program(cl_wrapper wrapper, const char *filename, cl_int *errcode);
cl_kernel cl_make_wrapper_kernel(cl_wrapper wrapper, cl_program program, const char *kername, cl_int *errcode);
void cl_destroy_wrapper(cl_wrapper wrapper);
void cl_get_platform_info(cl_wrapper wrapper, cl_platform_info param_name);
void cl_print_device_info(cl_wrapper wrapper, cl_device_info param_name);

#ifdef __cplusplus
}
#endif

#endif
#endif