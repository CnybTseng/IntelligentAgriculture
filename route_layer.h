#ifndef _ROUTE_LAYER_H_
#define _ROUTE_LAYER_H_

#include "convnet.h"

#ifdef __cplusplus
extern "C"
{
#endif

void forward_route_layer(route_layer *layer, convnet *net);
void backward_route_layer(route_layer *layer, convnet *net);

#ifdef __cplusplus
}
#endif

#endif