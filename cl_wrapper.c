#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <stdarg.h>
#include "cl_wrapper.h"

static cl_program cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename, cl_int *errcode);
static cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename, cl_int *errcode);
static cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename);
static void cl_host_mem_free(int n, ...);

cl_wrapper cl_create_wrapper(cl_int *errcode)
{
	cl_wrapper wrapper;
	
	const cl_uint num_entries = 1;
	*errcode = clGetPlatformIDs(num_entries, &wrapper.platform, NULL);
	if (CL_SUCCESS != *errcode) return wrapper;

	*errcode = clGetDeviceIDs(wrapper.platform, CL_DEVICE_TYPE_GPU, num_entries, &wrapper.device, NULL);
	if (CL_SUCCESS != *errcode) return wrapper;

	wrapper.context = clCreateContext(NULL, num_entries, &wrapper.device, NULL, NULL, errcode);
	if (!wrapper.context || CL_SUCCESS != *errcode) return wrapper;
	
	cl_command_queue_properties properties = 0;
#ifdef CL_PROFILING_ENABLE
	properties = CL_QUEUE_PROFILING_ENABLE;
#endif
	
	wrapper.command_queue = clCreateCommandQueue(wrapper.context, wrapper.device, properties, errcode);
	if (!wrapper.command_queue || CL_SUCCESS != *errcode) return wrapper;
	
	return wrapper;
}

cl_context cl_get_wrapper_context(cl_wrapper wrapper)
{
	return wrapper.context;
}

cl_command_queue cl_get_wrapper_command_queue(cl_wrapper wrapper)
{
	return wrapper.command_queue;
}

cl_program cl_make_wrapper_program(cl_wrapper wrapper, const char *filename, cl_int *errcode)
{
	char binary_filename[1024];
	strcpy(binary_filename, filename);
	strcat(binary_filename, ".bin");
	
	cl_program program = cl_create_program_from_binary(wrapper.device, wrapper.context, binary_filename, errcode);
	if (!program) {
		program = cl_create_program_with_source(wrapper.device, wrapper.context, filename, errcode);
		if (!program) return program;
		*errcode = cl_save_binary_program(wrapper.device, program, binary_filename);
	}
	
	return program;
}

cl_kernel cl_make_wrapper_kernel(cl_wrapper wrapper, cl_program program, const char *kername, cl_int *errcode)
{
	return clCreateKernel(program, kername, errcode);
}

void cl_destroy_wrapper(cl_wrapper wrapper)
{
	clReleaseCommandQueue(wrapper.command_queue);
	clReleaseContext(wrapper.context);
}

cl_program cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename, cl_int *errcode)
{		
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	char *strings = calloc(stbuf.st_size, sizeof(char));
	if (!strings) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(strings);
		return 0;
	}
	
	fread(strings, sizeof(char), stbuf.st_size, fp);
	fclose(fp);
	
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&strings, NULL, errcode);
	free(strings);
	
	if (!program || CL_SUCCESS != *errcode) return program;
	
	*errcode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (CL_SUCCESS != *errcode) {
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "clGetProgramBuildInfo:\n%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename, cl_int *errcode)
{	
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	unsigned char *binaries = calloc(stbuf.st_size, sizeof(unsigned char));
	if (!binaries) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(binaries);
		return 0;
	}
	
	const size_t length = stbuf.st_size;
	fread(binaries, sizeof(unsigned char), length, fp);
	fclose(fp);
	
	cl_int binary_status;
	cl_program program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **)&binaries,
		&binary_status, errcode);
	free(binaries);
	
	if (!program || CL_SUCCESS != *errcode) return program;
	
	*errcode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (CL_SUCCESS != *errcode) {
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "clGetProgramBuildInfo:\n%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename)
{
	cl_uint ndevices;
	cl_int errcode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &ndevices, NULL);
	if (CL_SUCCESS != errcode) return errcode;
	
	cl_device_id *devices = calloc(ndevices, sizeof(cl_device_id));
	if (!devices) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return CL_WRAPPER_CALLOC_FAIL;
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_DEVICES, ndevices * sizeof(cl_device_id), devices, NULL);
	if (CL_SUCCESS != errcode) {
		free(devices);
		return errcode;
	}
	
	size_t *sizes = calloc(ndevices, sizeof(size_t));
	if (!sizes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(devices);
		return CL_WRAPPER_CALLOC_FAIL;
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, ndevices * sizeof(size_t), sizes, NULL);
	if (CL_SUCCESS != errcode) {
		cl_host_mem_free(2, devices, sizes);
		return errcode;
	}
	
	unsigned char **binaries = calloc(ndevices, sizeof(unsigned char *));
	if (!binaries) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cl_host_mem_free(2, devices, sizes);
		return CL_WRAPPER_CALLOC_FAIL;
	}
		
	for (cl_uint i = 0; i < ndevices; i++) {
		binaries[i] = calloc(sizes[i], 1);
		if (!binaries[i]) {
			fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
			for (cl_uint j = 0; j < i; j++) free(binaries[j]);
			cl_host_mem_free(3, binaries, devices, sizes);
			return CL_WRAPPER_CALLOC_FAIL;
		}
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_BINARIES, ndevices * sizeof(unsigned char *), binaries, NULL);
	if (CL_SUCCESS != errcode) {
		for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
		cl_host_mem_free(3, binaries, devices, sizes);
		return errcode;
	}
	
	for (cl_uint i = 0; i < ndevices; i++) {
		if (devices[i] != device) continue;
		FILE *fp = fopen(filename, "wb");
		if (fp) {
			fwrite(binaries[i], sizeof(unsigned char), sizes[i], fp);
			fclose(fp);
		} else fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		break;
	}
	
	for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
	cl_host_mem_free(3, binaries, devices, sizes);
	
	return CL_SUCCESS;
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