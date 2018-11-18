#ifndef _YOLO_LAYER_H_
#define _YOLO_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void forward_yolo_layer(yolo_layer *layer, convnet *net);
void backward_yolo_layer(yolo_layer *layer, convnet *net);

#ifdef __cplusplus
}
#endif

#endif