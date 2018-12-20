/** @file aicore.h
 ** @brief 智慧农业核心模块
 ** @author 曾志伟
 ** @date 2018.11.16
 **/

/*
Copyright (C) 2018 Chengdu ZLT Technology Co., Ltd.
All rights reserved.

This file is part of the smart agriculture toolkit and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef _AICORE_H_
#define _AICORE_H_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
	#pragma message("=========================================> building dynamic library...")
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
	#pragma message("<========================================= linking dynamic library...")
  #endif
#endif

#define AIC_OK                0
#define AIC_ALLOCATE_FAIL    -1
#define AIC_FILE_NOT_EXIST   -2
#define AIC_PUSH_FAIL        -3
#define AIC_INIT_FAIL        -4

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief 初始化AICore模块.
 ** @return 如果初始化成功,返回AIC_OK.
 **         如果初始化失败,返回错误码.
 **/
DLL_PUBLIC int ai_core_init(void *);

/** @brief 将位图推送到AICore模块的队列缓冲区.
 ** @param bmp Bitmap格式图像数据.
 ** @param size 位图字节大小.size是位图文件头,信息头,调色板和图像数据大小总和.
 ** @return 如果推送成功,返回AIC_OK.
 **         如果推送失败,返回错误码.
 **/
DLL_PUBLIC int ai_core_push_image(const char *bmp, size_t size);

/** @brief 从AICore模块的队列中获取包含检测结果的位图.检测到的物体将以包围框标注.
 ** @param bmp Bitmap格式图像数据.
 ** @return 返回位图大小.在bmp不为空的情况下,如果获取成功,返回的位图大小等于位图文件头,信息头,调色板和
 **         图像数据大小总和,如果获取失败,返回0.在bmp为空的情况下,返回存储位图需要的缓冲区大小.
 **/
DLL_PUBLIC size_t ai_core_pull_image(char *bmp);

/** @brief 释放AICore模块所有资源.
 **/
DLL_PUBLIC void ai_core_free();

#ifdef __cplusplus
}
#endif

#endif