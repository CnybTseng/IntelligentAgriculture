#include <stdio.h>
#include <stdlib.h>
#include "aicore.h"

void test_aicore(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	test_aicore(argc, argv);
	
	return 0;
}

void test_aicore(int argc, char *argv[])
{
	int ret = ai_core_init();
	if (AIC_OK != ret) {
		fprintf(stderr, "ai_core_init[%s:%d:%d].\n", __FILE__, __LINE__, ret);
		return;
	}
	
	ai_core_free();
}