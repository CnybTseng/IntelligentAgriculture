ARCH=arm
GPU=1
NNPACK=0
CLBLAST=1

RM=rm
EXE_SUFFIX=

CS=$(wildcard *.c)
CNDS=$(notdir $(CS))
COBJS=$(patsubst %.c,%.o,$(CNDS))

CPPS=$(wildcard *.cpp)
CPPNDS=$(notdir $(CPPS))
CPPOBJS=$(patsubst %.cpp,%.o,$(CPPNDS))

EXEOBJ=test_znet.o test_aicore.o
ALLOBJS=$(COBJS) $(CPPOBJS)
OBJS=$(filter-out $(EXEOBJ),$(ALLOBJS))

SLIB=libaicore.so
ALIB=libaicore.a

_EXEC=test_znet test_aicore
ifeq ($(ARCH),x86)
EXE_SUFFIX=.exe
EXEC=$(addsuffix $(EXE_SUFFIX),$(_EXEC))
else ifeq ($(ARCH),arm)
EXEC=$(_EXEC)
endif

ifeq ($(ARCH),x86)
CC=gcc
else ifeq ($(ARCH),arm)
CC=$(ANDROID_TOOLCHAIN_PATH)/bin/arm-linux-androideabi-gcc
endif

ifeq ($(ARCH),x86)
AR=ar
ARFLAGS=rcs
else ifeq ($(ARCH),arm)
AR=$(ANDROID_TOOLCHAIN_PATH)/arm-linux-androideabi/bin/ar
ARFLAGS=rcs
endif

INC=
ifeq ($(ARCH),x86)
INC+= -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include" \
-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include/CL" \
-I../thirdparty/pthreads-2.9.1/include
else ifeq ($(ARCH),arm)
INC+= -I../thirdparty/opencl-1.1/include -I../thirdparty/NNPACK/include \
-I../thirdparty/clblast/include
endif

CFLAGS=$(INC) -Wall -fPIC -O3 -DCL_TARGET_OPENCL_VERSION=110 -g  -fopenmp -DMERGE_BATCHNORM_TO_CONV
ifeq ($(GPU),1)
CFLAGS+= -DOPENCL
ifeq ($(CLBLAST),1)
CFLAGS+= -DCLBLAST
endif
else ifeq ($(NNPACK),1)
CFLAGS+= -DNNPACK
endif
ifeq ($(ARCH),x86)
CFLAGS+= -msse2 -mssse3 -D__INTEL_SSE__ -D_TIMESPEC_DEFINED
else ifeq ($(ARCH),arm)
CFLAGS+= -march=armv7-a -mfloat-abi=softfp -mfpu=neon -std=c99 -D__ANDROID_API__=24 -pie -fPIE
endif

LIB= -L./
LIBS=
ifeq ($(ARCH),x86)
LIB+= -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/Win32" \
-L../thirdparty/pthreads-2.9.1/lib/x86
LIBS+= -lpthread
else ifeq ($(ARCH),arm)
LIB+= -L../thirdparty/opencl-1.1/lib/armeabi-v7a -L../thirdparty/NNPACK/lib \
-L../thirdparty/clblast/lib
LIBS+= -lm
endif
ifeq ($(GPU),1)
LIBS+= -lOpenCL
ifeq ($(CLBLAST),1)
LIBS+= -lclblast
endif
else ifeq ($(NNPACK),1)
LIBS+= -lpthreadpool -lnnpack -lcpuinfo -lclog -llog
endif

LDFLAGS=$(LIB) $(LIBS)
ifeq ($(ARCH),arm)
LDFLAGS+= -march=armv7-a -Wl,--fix-cortex-a8
endif

.PHONY:$(EXEC) all
all:info $(SLIB) $(EXEC)

test_znet$(EXE_SUFFIX):test_znet.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_aicore$(EXE_SUFFIX):test_aicore.o bitmap.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -laicore
	
$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

%.o:%.c
	$(CC) $(CFLAGS) -c $^

info:
	@echo objects:$(ALLOBJS)
	
.PHONY:clean
clean:
	$(RM) $(ALLOBJS) $(EXEC) $(SLIB)