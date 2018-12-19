#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "aicore.h"
#include "bitmap.h"

void print_help();
void test_aicore(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	if (argc < 2){
		print_help();
		return 0;
	}
	
	test_aicore(argc, argv);
	
	return 0;
}

void print_help()
{
	printf("Usage:\n");
	printf("      test_aicore [bitmap filename]\n");
}

void test_aicore(int argc, char *argv[])
{
	int ret = ai_core_init(NULL);
	ret = ai_core_init(NULL);
	if (AIC_OK != ret) {
		fprintf(stderr, "ai_core_init[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		return ai_core_free();
	}

	struct stat st_buffer;
	if (stat(argv[1], &st_buffer)) {
		fprintf(stderr, "stat[%s:%d:%d]\n", __FILE__, __LINE__, errno);
		return ai_core_free();
	}
	
	bitmap *send_buffer = read_bmp(argv[1]);
	if (!send_buffer) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return ai_core_free();
	}
	
	int status = ai_core_push_image((char *)send_buffer, st_buffer.st_size);
	if (status != AIC_OK) {
		fprintf(stderr, "ai_core_push_image[%s:%d:%d]\n", __FILE__, __LINE__, status);
		delete_bmp(send_buffer);
		return ai_core_free();
	}

	char *receive_buffer = calloc(st_buffer.st_size, sizeof(char));
	if (!receive_buffer) {
		fprintf(stderr, "calloc[%s:%d]\n", __FILE__, __LINE__);
		delete_bmp(send_buffer);
		return ai_core_free();
	}
	
	size_t size = ai_core_pull_image(receive_buffer);
	if (size != st_buffer.st_size) {
		fprintf(stderr, "ai_core_pull_image[%s:%d]\n", __FILE__, __LINE__);
	}
	
	delete_bmp(send_buffer);
	if (receive_buffer) free(receive_buffer);
	
	ai_core_free();
}