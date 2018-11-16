/** @file aicore.h
 ** @brief 智慧农业核心模块
 ** @author 曾志伟
 ** @date 2018.11.16
 **/

/*
Copyright (C) 2018 Chengdu ZLT Technology Co., Ltd.
All rights reserved.

This file is part of the intellectual agriculture toolkit and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef _AICORE_H_
#define _AICORE_H_

#define AIC_OK                0
#define AIC_ALLOCATE_FAIL    -1
#define AIC_FILE_NOT_EXIST   -2
#define AIC_PUSH_FAIL        -3

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief 初始化AICore模块.
 ** @return 如果初始化成功,返回AIC_OK.
 **         如果初始化失败,返回错误码.
 **/
int ai_core_init();

/** @brief 将位图推送到AICore模块的队列缓冲区.
 ** @param bmp Bitmap格式图像数据.
 ** @param size 位图字节大小.size是位图文件头,信息头,调色板和图像数据大小总和.
 ** @return 如果推送成功,返回AIC_OK.
 **         如果推送失败,返回错误码.
 **/
int ai_core_push_image(const char *bmp, size_t size);

/** @brief 从AICore模块的队列中获取包含检测结果的位图.检测到的物体将以包围框标注.
 ** @param bmp Bitmap格式图像数据.
 ** @return 返回位图大小.在bmp不为空的情况下,如果获取成功,返回的位图大小等于位图文件头,信息头,调色板和
 **         图像数据大小总和,如果获取失败,返回0.在bmp为空的情况下,返回存储位图需要的缓冲区大小.
 **/
size_t ai_core_pull_image(char *bmp);

/** @brief 释放AICore模块所有资源.
 **/
void ai_core_free();

#ifdef __cplusplus
}
#endif

#endif