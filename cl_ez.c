/** @file cl_ez.c - Implementation
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdarg.h>

#include "cl_ez.h"

/** @name Some local methods.
 ** @{ */
static cl_platform_id *cl_get_platform_ids(cl_uint *nplatforms);
static cl_device_id *cl_get_device_ids(cl_platform_id platform, cl_device_type device_type, cl_uint *ndevices);
static cl_program cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename);
static cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename);
static cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename);
static cl_uint cl_get_device_max_compute_units(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_work_item_dimensions(cl_platform_layer *platform_layer);
static size_t *cl_get_device_max_work_item_sizes(cl_platform_layer *platform_layer, cl_uint max_work_item_dimensions);
static size_t cl_get_device_max_work_group_size(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_clock_frequency(cl_platform_layer *platform_layer);
static cl_ulong cl_get_device_max_mem_alloc_size(cl_platform_layer *platform_layer);
static cl_bool cl_get_device_image_support(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_read_image_args(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_write_image_args(cl_platform_layer *platform_layer);
static size_t cl_get_device_image2d_max_width(cl_platform_layer *platform_layer);
static size_t cl_get_device_image2d_max_height(cl_platform_layer *platform_layer);
static size_t cl_get_device_image3d_max_width(cl_platform_layer *platform_layer);
static size_t cl_get_device_image3d_max_height(cl_platform_layer *platform_layer);
static size_t cl_get_device_image3d_max_depth(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_samplers(cl_platform_layer *platform_layer);
static size_t cl_get_device_max_parameter_size(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_global_mem_cacheline_size(cl_platform_layer *platform_layer);
static cl_ulong cl_get_device_global_mem_cache_size(cl_platform_layer *platform_layer);
static cl_ulong cl_get_device_global_mem_size(cl_platform_layer *platform_layer);
static cl_ulong cl_get_device_max_constant_buffer_size(cl_platform_layer *platform_layer);
static cl_uint cl_get_device_max_constant_args(cl_platform_layer *platform_layer);
static cl_ulong cl_get_device_local_mem_size(cl_platform_layer *platform_layer);
static cl_bool cl_get_device_available(cl_platform_layer *platform_layer);
static cl_bool cl_get_device_compiler_available(cl_platform_layer *platform_layer);
static void cl_host_mem_free(int n, ...);
/** @} */

/** @brief Create a OpenCL platform layer.
 ** @return a valid non-zero OpenCL platform layer if success,
 **         NULL if fail.
 **/
cl_platform_layer *cl_create_platform_layer()
{
	cl_platform_layer *platform_layer = (cl_platform_layer *)malloc(sizeof(cl_platform_layer));
	if (!platform_layer) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return platform_layer;
	}
	
	platform_layer->platforms = NULL;
	platform_layer->nplatforms = 0;
	platform_layer->devices = NULL;
	platform_layer->ndevices = 0;
	platform_layer->context = 0;
	
	platform_layer->platforms = cl_get_platform_ids(&platform_layer->nplatforms);
	if (!platform_layer->platforms) goto clean;
	
	platform_layer->devices = cl_get_device_ids(platform_layer->platforms[0], CL_DEVICE_TYPE_GPU,
	&platform_layer->ndevices);
	if (!platform_layer->devices) goto clean;
	
	cl_context_properties properties[] = {(cl_context_properties)CL_CONTEXT_PLATFORM,
	(cl_context_properties)platform_layer->platforms[0], 0};
	
	cl_int erret;
	platform_layer->context = clCreateContext(properties, 1, &platform_layer->devices[0],
	NULL, NULL, &erret);
	if (!platform_layer->context || CL_SUCCESS != erret) {
		fprintf(stderr, "clCreateContext[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		clean:cl_destroy_platform_layer(platform_layer);
	}
		
	return platform_layer;
}

/** @brief Destroy a OpenCL platform layer.
 ** @param platform_layer a OpenCL platform layer.
 **/
void cl_destroy_platform_layer(cl_platform_layer *platform_layer)
{
	if (!platform_layer) return;
	
	if (platform_layer->platforms) {
		free(platform_layer->platforms);
		platform_layer->platforms = NULL;
		platform_layer->nplatforms = 0;
	}
	
	if (platform_layer->devices) {
		free(platform_layer->devices);
		platform_layer->devices = NULL;
		platform_layer->ndevices = 0;
	}
	
	if (platform_layer->context) {
		cl_int ret = clReleaseContext(platform_layer->context);
		if (CL_SUCCESS != ret)
			fprintf(stderr, "clReleaseContext[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	free(platform_layer);
	platform_layer = NULL;
}

/** @brief Query the list of OpenCL platforms available.
 ** @param nplatforms the number of OpenCL platforms.
 ** @return the list of OpenCL platforms.
 **/
cl_platform_id *cl_get_platform_ids(cl_uint *nplatforms)
{
	*nplatforms = 0;
	cl_platform_id *platforms = NULL;
	
	cl_int ret = clGetPlatformIDs(0, NULL, nplatforms);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetPlatformIDs[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		return platforms;
	}
	
	platforms = (cl_platform_id *)malloc((*nplatforms) * sizeof(cl_platform_id));
	if (!platforms) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return platforms;
	}
	
	ret = clGetPlatformIDs(*nplatforms, platforms, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetPlatformIDs[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		*nplatforms = 0;
		free(platforms);
		platforms = NULL;
	}
	
	return platforms;
}

/** @brief Query the list of devices available on a platform.
 ** @param platform the platform ID.
 ** @param device_type the type of OpenCL device.
 ** @param ndevices the number of OpenCL devices.
 ** @return the list of OpenCL devices.
 **/
cl_device_id *cl_get_device_ids(cl_platform_id platform, cl_device_type device_type, cl_uint *ndevices)
{
	*ndevices = 0;
	cl_device_id *devices = NULL;
	
	cl_int ret = clGetDeviceIDs(platform, device_type, 0, NULL, ndevices);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceIDs[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		return devices;
	}
	
	devices = (cl_device_id *)malloc((*ndevices) * sizeof(cl_device_id));
	if (!devices) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return devices;
	}
	
	ret = clGetDeviceIDs(platform, device_type, *ndevices, devices, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceIDs[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		*ndevices = 0;
		free(devices);
		devices = NULL;
	}
	
	return devices;
}

/** @brief Query OpenCL platform layer information.
 ** @param platform_layer a OpenCL platform layer.
 **/
void cl_get_platform_layer_info(cl_platform_layer *platform_layer)
{
	printf("number of platforms is %d.\n", platform_layer->nplatforms);
	printf("number of devices of the 1st device is %d.\n", platform_layer->ndevices);
	
	cl_uint max_compute_units = cl_get_device_max_compute_units(platform_layer);
	printf("maximum compute units is %d.\n", max_compute_units);
	
	cl_uint max_work_item_dimensions = cl_get_device_max_work_item_dimensions(platform_layer);
	printf("maximum work item dimensions is %d.\n", max_work_item_dimensions);
	
	size_t *max_work_item_sizes = cl_get_device_max_work_item_sizes(platform_layer, max_work_item_dimensions);
	if (max_work_item_sizes) {
		printf("maximum work item sizes is ");
		for (cl_uint i = 0; i < max_work_item_dimensions; i++) {
			printf("%lu", (unsigned long)max_work_item_sizes[i]);
			if (i < max_work_item_dimensions - 1) printf("x");
			else printf("\n");
		}
		
		free(max_work_item_sizes);
	}
	
	size_t max_work_group_size = cl_get_device_max_work_group_size(platform_layer);
	printf("maximum work group size is %lu.\n", (unsigned long)max_work_group_size);
	
	cl_uint max_clock_frequency = cl_get_device_max_clock_frequency(platform_layer);
	printf("maximum clock frequency is %d MHz.\n", max_clock_frequency);
	
	cl_ulong max_mem_alloc_size = cl_get_device_max_mem_alloc_size(platform_layer);
	printf("maximum memory allocation size is %lu bytes.\n", (unsigned long)max_mem_alloc_size);
	
	cl_bool image_support = cl_get_device_image_support(platform_layer);
	if (CL_TRUE == image_support) printf("images are supported by the OpenCL device.\n");
	else printf("images are not supported by the OpenCL device.\n");
	
	cl_uint max_read_image_args = cl_get_device_max_read_image_args(platform_layer);
	printf("maximum number of simultaneous image that can be read is %u.\n", max_read_image_args);
	
	cl_uint max_write_image_args = cl_get_device_max_write_image_args(platform_layer);
	printf("maximum number of simultaneous image that can be write is %u.\n", max_write_image_args);
	
	size_t image2d_max_width = cl_get_device_image2d_max_width(platform_layer);
	printf("maximum width of 2D image is %lu.\n", (unsigned long)image2d_max_width);
	
	size_t image2d_max_height = cl_get_device_image2d_max_height(platform_layer);
	printf("maximum height of 2D image is %lu.\n", (unsigned long)image2d_max_height);
	
	size_t image3d_max_width = cl_get_device_image3d_max_width(platform_layer);
	printf("maximum width of 3D image is %lu.\n", (unsigned long)image3d_max_width);
	
	size_t image3d_max_height = cl_get_device_image3d_max_height(platform_layer);
	printf("maximum height of 3D image is %lu.\n", (unsigned long)image3d_max_height);
	
	size_t image3d_max_depth = cl_get_device_image3d_max_depth(platform_layer);
	printf("maximum depth of 3D image is %lu.\n", (unsigned long)image3d_max_depth);
	
	cl_uint max_samplers = cl_get_device_max_samplers(platform_layer);	
	printf("maximum number of sampler is %u.\n", max_samplers);
	
	size_t max_parameter_size = cl_get_device_max_parameter_size(platform_layer);	
	printf("maximum parameter size is %lu bytes.\n", (unsigned long)max_parameter_size);
	
	cl_uint global_mem_cacheline_size = cl_get_device_global_mem_cacheline_size(platform_layer);
	printf("global memory cacheline size is %u bytes.\n", global_mem_cacheline_size);
	
	cl_ulong global_mem_cache_size = cl_get_device_global_mem_cache_size(platform_layer);
	printf("global memory cache size is %lu bytes.\n", (unsigned long)global_mem_cache_size);
	
	cl_ulong global_mem_size = cl_get_device_global_mem_size(platform_layer);
	printf("global memory size is %lu bytes.\n", (unsigned long)global_mem_size);
	
	cl_ulong max_constant_buffer_size = cl_get_device_max_constant_buffer_size(platform_layer);
	printf("maximum constant buffer size is %lu bytes.\n", (unsigned long)max_constant_buffer_size);
	
	cl_uint max_constant_args = cl_get_device_max_constant_args(platform_layer);
	printf("maximum number of constant arguments declared in a kernel is %u.\n", max_constant_args);
	
	cl_ulong local_mem_size = cl_get_device_local_mem_size(platform_layer);
	printf("local memory size is %lu bytes.\n", (unsigned long)local_mem_size);
	
	cl_bool available = cl_get_device_available(platform_layer);
	if (CL_TRUE == available) printf("device is available.\n");
	else printf("device is not available.\n");
	
	cl_bool compiler_available = cl_get_device_compiler_available(platform_layer);
	if (CL_TRUE == compiler_available) printf("device compiler is available.\n");
	else printf("device compiler is not available.\n");
}

/** @brief Create a OpenCL runtime.
 ** @param device a OpenCL device.
 ** @param context a OpenCL context.
 ** @param filename kernel filename.
 ** @return a valid non-zero OpenCL runtime if success,
 **         NULL if fail.
 **/
cl_runtime *cl_create_runtime(cl_device_id device, cl_context context, const char *filename)
{
	cl_runtime *runtime = (cl_runtime *)malloc(sizeof(cl_runtime));
	if (!runtime) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return runtime;
	}
	
	runtime->cmdqueue = 0;
	runtime->program = 0;
	runtime->kernel = 0;
	
	cl_int erret;
	runtime->cmdqueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &erret);
	if (!runtime->cmdqueue || CL_SUCCESS != erret) {
		fprintf(stderr, "clCreateCommandQueue[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		goto clean;
	}
	
	char binary_filename[1024];
	strcpy(binary_filename, filename);
	strcat(binary_filename, ".bin");
	
	runtime->program = cl_create_program_from_binary(device, context, binary_filename);
	if (!runtime->program) {
		runtime->program = cl_create_program_with_source(device, context, filename);
		if (!runtime->program) goto clean;
		if (cl_save_binary_program(device, runtime->program, binary_filename)) goto clean;
	}
	
	char kernel_name[256];
	sscanf(filename, "%[^.]", kernel_name);
	
	runtime->kernel = clCreateKernel(runtime->program, (const char *)kernel_name, &erret);
	if (!runtime->kernel || CL_SUCCESS != erret) {
		fprintf(stderr, "clCreateKernel[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		clean:cl_destroy_runtime(runtime);
	}
		
	return runtime;
}

/** @brief Destroy a OpenCL runtime.
 ** @param runtime a OpenCL runtime.
 **/
void cl_destroy_runtime(cl_runtime *runtime)
{
	if (!runtime) return;
	
	if (runtime->cmdqueue) {
		cl_int ret = clReleaseCommandQueue(runtime->cmdqueue);
		if (CL_SUCCESS != ret)
			fprintf(stderr, "clReleaseCommandQueue[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	if (runtime->program) {
		cl_int ret = clReleaseProgram(runtime->program);
		if (CL_SUCCESS != ret)
			fprintf(stderr, "clReleaseProgram[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	if (runtime->kernel) {
		cl_int ret = clReleaseKernel(runtime->kernel);
		if (CL_SUCCESS != ret)
			fprintf(stderr, "clReleaseKernel[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	free(runtime);
	runtime = NULL;
}

/** @brief Creates a program object for a context, and loads the source code specified by the text
 **        strings in the strings array into the program object. Then builds (compiles & links) a
 **        program executable from the program source for all the devices or a specific device(s) in
 **        the OpenCL context associated with program.
 ** @param device a OpenCL device.
 ** @param context a valid OpenCL context.
 ** @param filename kernel filename.
 ** @return a valid non-zero program object if success,
 **         NULL if fail.
 **/
cl_program cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename)
{
	cl_program program = 0;
		
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return program;
	}
	
	char *strings = (char *)malloc(stbuf.st_size * sizeof(char));
	if (!strings) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return program;
	}
	
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(strings);
		return program;
	}
	
	fread(strings, sizeof(char), stbuf.st_size, fp);
	fclose(fp);
	
	cl_int erret;
	program = clCreateProgramWithSource(context, 1, (const char **)&strings, NULL, &erret);
	free(strings);
	if (!program || CL_SUCCESS != erret) {
		fprintf(stderr, "clCreateProgramWithSource[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		return program;
	}
	
	erret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clBuildProgram[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

/** @brief creates a program object for a context, and loads the binary bits specified by binary into the
 **        program object. Then builds (compiles & links) a program executable from the program source
 **        for all the devices or a specific device(s) in the OpenCL context associated with program.
 ** @param device a OpenCL device.
 ** @param context a valid OpenCL context.
 ** @param filename kernel binary filename.
 ** @return a valid non-zero program object if success,
 **         NULL if fail.
 **/
cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename)
{
	cl_program program = 0;
	
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return program;
	}
	
	unsigned char *binaries = (unsigned char *)malloc(sizeof(unsigned char) * stbuf.st_size);
	if (!binaries) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return program;
	}
	
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(binaries);
		return program;
	}
	
	const size_t length = stbuf.st_size;
	fread(binaries, sizeof(unsigned char), length, fp);
	fclose(fp);
	
	cl_int binary_status;
	cl_int erret;
	program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **)&binaries,
	&binary_status, &erret);
	free(binaries);
	if (!program || CL_SUCCESS != erret) {
		fprintf(stderr, "clCreateProgramWithBinary[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		return program;
	}
	
	erret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clBuildProgram[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

/** @brief Save kernel as binary file.
 ** @param device a OpenCL device.
 ** @param program a valid OpenCL context.
 ** @param filename kernel binary filename.
 ** @return  0 if success,
 **         -1 if fail.
 **/
cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename)
{
	cl_uint ndevices;
	cl_int erret = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &ndevices, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clGetProgramInfo[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		return -1;
	}
	
	cl_device_id *devices = (cl_device_id *)malloc(ndevices * sizeof(cl_device_id));
	if (!devices) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	erret = clGetProgramInfo(program, CL_PROGRAM_DEVICES, ndevices * sizeof(cl_device_id), devices, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clGetProgramInfo[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		free(devices);
		return -1;
	}
	
	size_t *sizes = (size_t *)malloc(ndevices * sizeof(size_t));
	if (!sizes) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		free(devices);
		return -1;
	}
	
	erret = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, ndevices * sizeof(size_t), sizes, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clGetProgramInfo[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		cl_host_mem_free(2, devices, sizes);
		return -1;
	}
	
	unsigned char **binaries = (unsigned char **)malloc(ndevices * sizeof(unsigned char *));
	if (!binaries) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		cl_host_mem_free(2, devices, sizes);
		return -1;
	}
		
	for (cl_uint i = 0; i < ndevices; i++) {
		binaries[i] = (unsigned char *)malloc(sizes[i]);
		if (!binaries[i]) {
			fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
			for (cl_uint j = 0; j < i; j++) free(binaries[j]);
			cl_host_mem_free(3, binaries, devices, sizes);
			return -1;
		}
	}
	
	erret = clGetProgramInfo(program, CL_PROGRAM_BINARIES, ndevices * sizeof(unsigned char *), binaries, NULL);
	if (CL_SUCCESS != erret) {
		fprintf(stderr, "clGetProgramInfo[%s:%d:%d].\n", __FILE__, __LINE__, erret);
		for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
		cl_host_mem_free(3, binaries, devices, sizes);
		return -1;
	}
	
	for (cl_uint i = 0; i < ndevices; i++) {
		if (devices[i] != device) continue;
		FILE *fp = fopen(filename, "wb");
		if (fp) {
			fwrite(binaries[i], sizeof(unsigned char), sizes[i], fp);
			fclose(fp);
		} else {
			fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		}
		
		break;
	}
	
	for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
	cl_host_mem_free(3, binaries, devices, sizes);
	
	return 0;
}

cl_uint cl_get_device_max_compute_units(cl_platform_layer *platform_layer)
{
	cl_uint max_compute_units;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
	&max_compute_units, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_compute_units;
}

cl_uint cl_get_device_max_work_item_dimensions(cl_platform_layer *platform_layer)
{
	cl_uint max_work_item_dimensions;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
	&max_work_item_dimensions, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_work_item_dimensions;
}

size_t *cl_get_device_max_work_item_sizes(cl_platform_layer *platform_layer, cl_uint max_work_item_dimensions)
{
	size_t *max_work_item_sizes = (size_t *)calloc(max_work_item_dimensions, sizeof(size_t));
	if (!max_work_item_sizes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
	} else {
		cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
			max_work_item_dimensions * sizeof(size_t), max_work_item_sizes, NULL);
		if (CL_SUCCESS != ret) {
			fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
			free(max_work_item_sizes);
			max_work_item_sizes = NULL;
		}
	}
	
	return max_work_item_sizes;
}

size_t cl_get_device_max_work_group_size(cl_platform_layer *platform_layer)
{
	size_t max_work_group_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
	&max_work_group_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_work_group_size;
}

cl_uint cl_get_device_max_clock_frequency(cl_platform_layer *platform_layer)
{
	cl_uint max_clock_frequency;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
	&max_clock_frequency, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_clock_frequency;
}

cl_ulong cl_get_device_max_mem_alloc_size(cl_platform_layer *platform_layer)
{
	cl_ulong max_mem_alloc_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
	&max_mem_alloc_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_mem_alloc_size;
}

cl_bool cl_get_device_image_support(cl_platform_layer *platform_layer)
{
	cl_bool image_support;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
	&image_support, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image_support;
}

cl_uint cl_get_device_max_read_image_args(cl_platform_layer *platform_layer)
{
	cl_uint max_read_image_args;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(cl_uint),
	&max_read_image_args, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_read_image_args;
}

cl_uint cl_get_device_max_write_image_args(cl_platform_layer *platform_layer)
{
	cl_uint max_write_image_args;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(cl_uint),
	&max_write_image_args, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_write_image_args;
}

size_t cl_get_device_image2d_max_width(cl_platform_layer *platform_layer)
{
	size_t image2d_max_width;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t),
	&image2d_max_width, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image2d_max_width;
}

size_t cl_get_device_image2d_max_height(cl_platform_layer *platform_layer)
{
	size_t image2d_max_height;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t),
	&image2d_max_height, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image2d_max_height;
}

size_t cl_get_device_image3d_max_width(cl_platform_layer *platform_layer)
{
	size_t image3d_max_width;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t),
	&image3d_max_width, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image3d_max_width;
}

size_t cl_get_device_image3d_max_height(cl_platform_layer *platform_layer)
{
	size_t image3d_max_height;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t),
	&image3d_max_height, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image3d_max_height;
}

size_t cl_get_device_image3d_max_depth(cl_platform_layer *platform_layer)
{
	size_t image3d_max_depth;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t),
	&image3d_max_depth, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return image3d_max_depth;
}

cl_uint cl_get_device_max_samplers(cl_platform_layer *platform_layer)
{
	cl_uint max_samplers;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint),
	&max_samplers, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_samplers;
}

size_t cl_get_device_max_parameter_size(cl_platform_layer *platform_layer)
{
	size_t max_parameter_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t),
	&max_parameter_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_parameter_size;
}

cl_uint cl_get_device_global_mem_cacheline_size(cl_platform_layer *platform_layer)
{
	cl_uint global_mem_cacheline_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
	sizeof(cl_uint), &global_mem_cacheline_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return global_mem_cacheline_size;
}

cl_ulong cl_get_device_global_mem_cache_size(cl_platform_layer *platform_layer)
{
	cl_ulong global_mem_cache_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
	sizeof(cl_ulong), &global_mem_cache_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return global_mem_cache_size;
}

cl_ulong cl_get_device_global_mem_size(cl_platform_layer *platform_layer)
{
	cl_ulong global_mem_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_GLOBAL_MEM_SIZE,
	sizeof(cl_ulong), &global_mem_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return global_mem_size;
}

cl_ulong cl_get_device_max_constant_buffer_size(cl_platform_layer *platform_layer)
{
	cl_ulong max_constant_buffer_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
	sizeof(cl_ulong), &max_constant_buffer_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_constant_buffer_size;
}

cl_uint cl_get_device_max_constant_args(cl_platform_layer *platform_layer)
{
	cl_uint max_constant_args;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint),
	&max_constant_args, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return max_constant_args;
}

cl_ulong cl_get_device_local_mem_size(cl_platform_layer *platform_layer)
{
	cl_ulong local_mem_size;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
	&local_mem_size, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return local_mem_size;
}

cl_bool cl_get_device_available(cl_platform_layer *platform_layer)
{
	cl_bool available;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_AVAILABLE, sizeof(cl_bool),
	&available, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return available;
}

cl_bool cl_get_device_compiler_available(cl_platform_layer *platform_layer)
{
	cl_bool compiler_available;
	cl_int ret = clGetDeviceInfo(platform_layer->devices[0], CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool),
	&compiler_available, NULL);
	if (CL_SUCCESS != ret) {
		fprintf(stderr, "clGetDeviceInfo[%s:%d:%d].\n", __FILE__, __LINE__, ret);
	}
	
	return compiler_available;
}

void cl_host_mem_free(int n, ...)
{
	va_list ap;
	va_start(ap, n);
	
	for (int i = 0; i < n; ++i) {
		void *p = va_arg(ap, void *);
		if (p) {
			free(p);
			p = NULL;
		}
	}
	
	va_end(ap);
}