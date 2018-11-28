X86=0
ARM=1

RM=rm
EXE_SUFFIX=

CS=$(wildcard *.c)
CNDS=$(notdir $(CS))
COBJS=$(patsubst %.c,%.o,$(CNDS))

CPPS=$(wildcard *.cpp)
CPPNDS=$(notdir $(CPPS))
CPPOBJS=$(patsubst %.cpp,%.o,$(CPPNDS))

EXEOBJ=detector.o testaic.o
ALLOBJS=$(COBJS) $(CPPOBJS)
OBJS=$(filter-out $(EXEOBJ),$(ALLOBJS))

SLIB=libaicore.so
ALIB=libaicore.a
_EXEC=detector testaic
ifeq ($(X86),1)
EXE_SUFFIX=.exe
EXEC=$(addsuffix $(EXE_SUFFIX),$(_EXEC))
else
EXEC=$(_EXEC)
endif

ifeq ($(X86),1)
CC=gcc
endif

ifeq ($(ARM),1)
CC=arm-linux-androideabi-gcc
endif

AR=ar
ARFLAGS=rcs

INC=
ifeq ($(X86),1)
INC+= -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include" \
-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include/CL"
endif
ifeq ($(ARM),1)
INC+= -I../thirdparty/opencl-1.1/include
endif

CFLAGS=$(INC) -Wall -fPIC -O3 -DCL_TARGET_OPENCL_VERSION=110 -g  -fopenmp
ifeq ($(X86),1)
CFLAGS+= -msse2 -mssse3 -D__INTEL_SSE__
endif
ifeq ($(ARM),1)
CFLAGS+= -march=armv7-a -mfloat-abi=softfp -mfpu=neon -std=c99 -D__ANDROID_API__=17
endif

LIB=
LIBS=
ifeq ($(X86),1)
LIB+= -L./ -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/Win32" \
-L../thirdparty/pthreads-2.9.1/lib/x86
LIBS+= -lOpenCL -lpthread
endif
ifeq ($(ARM),1)
LIB+= -L./ -L../thirdparty/opencl-1.1/lib/armeabi-v7a
LIBS+= -lm
endif

LDFLAGS=$(LIB) $(LIBS)
ifeq ($(ARM),1)
LDFLAGS+= -march=armv7-a -Wl,--fix-cortex-a8
endif

.PHONY:$(EXEC) all
all:info $(SLIB) $(ALIB) $(EXEC)

detector$(EXE_SUFFIX):detector.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

testaic$(EXE_SUFFIX):testaic.o
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
	$(RM) $(ALLOBJS) $(EXEC) $(SLIB) $(ALIB)