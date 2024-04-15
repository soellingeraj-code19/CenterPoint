OUTNAME_RELEASE = centerpoint
OUTNAME_DEBUG   = centerpoint_debug
EXTRA_DIRECTORIES = ../common
.NOTPARALLEL:
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)
CXXFLAGS += -I$(CUDA_INCLUDE_DIR) -I$(TRT_INCLUDE_DIR)
LDFLAGS += -L$(CUDA_LIB_DIR) -lcudart -lcublas -L$(TRT_LIB_DIR) -lnvinfer -lnvinfer_plugin

