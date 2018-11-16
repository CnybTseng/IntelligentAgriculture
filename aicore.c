#include <stdio.h>
#include "aicore.h"

int ai_core_init()
{
	printf("call ai_core_init\n");
	return AIC_OK;
}

int ai_core_push_image(const char *bmp, size_t size)
{
	printf("call ai_core_push_image\n");
	return AIC_OK;
}

size_t ai_core_pull_image(char *bmp)
{
	printf("call ai_core_pull_image\n");
	return 0;
}

void ai_core_free()
{
	printf("call ai_core_free\n");
}