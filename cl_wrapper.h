#ifdef OPENCL
#ifndef _CL_WRAPPER_H_
#define _CL_WRAPPER_H_

#ifdef __linux__
#	include <linux/ion.h>
#	include "CL/cl_ext_qcom.h"
#endif
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
#ifdef __linux__
	int ion_device_fd;
#endif
} cl_wrapper;

#ifdef __linux__
typedef struct {
	struct ion_allocation_data allocation_data;
	struct ion_handle_data handle_data;
    struct ion_fd_data fd_data;
	cl_mem_ion_host_ptr ion_mem;
} cl_ion_context;
#endif

cl_wrapper cl_create_wrapper(cl_int *errcode);
cl_program cl_make_wrapper_program(cl_wrapper wrapper, const char *filename, char *buffer, const char *options, cl_int *errcode);
cl_kernel cl_make_wrapper_kernel(cl_wrapper wrapper, cl_program program, const char *kername, cl_int *errcode);
void cl_destroy_wrapper(cl_wrapper wrapper);
void cl_print_platform_info(cl_wrapper wrapper, cl_platform_info param_name);
void cl_print_device_info(cl_wrapper wrapper, cl_device_info param_name);
#ifdef __linux__
size_t cl_get_ion_image_row_pitch(cl_wrapper wrapper, cl_image_format image_format, cl_image_desc image_desc);
cl_ion_context cl_make_ion_buffer_for_nonplanar_image(cl_wrapper wrapper, cl_image_desc image_desc);
void cl_free_ion_context(cl_wrapper wrapper, cl_ion_context ion_context);
#endif

#ifdef __cplusplus
}
#endif

#endif
#endif