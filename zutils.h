#ifndef _ZUTILS_H_
#define _ZUTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

void mmfree(int n, ...);
void mset(void *X, int size, const char *type, void *value);

#ifdef __cplusplus
}
#endif

#endif