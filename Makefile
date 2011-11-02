# Add source files here
EXECUTABLE	:= matyka_particle
# Cuda source files (compiled with cudacc)
CUFILES		:= kernels.cu \

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp \

USEGLLIB	     := 1
USEPARAMGL	     := 1
USEGLUT		     := 1
USECUFFT         := 1
#USECUDPP             := 1
USERENDERCHECKGL     := 1
USENEWINTEROP        := 1

include ../../common/common.mk
