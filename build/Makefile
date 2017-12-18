# operating system
HOST_OS := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# architecture
ARCH := $(shell getconf LONG_BIT)

NVCC := nvcc

ifeq ($(HOST_OS),darwin)
	CXX := clang++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS := -L$(CUDA_PATH)/lib
else
	CXX := g++
	CUDA_PATH ?= /usr/local/cuda
	LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
	LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64
	LIB_FLAGS := $(LIB_FLAGS_$(ARCH))
endif

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = ../googletest
SRC_DIR = ../src
INCLUDE_DIR = ../include
TEST_DIR = ../test
CUDA_DIR := /usr/local/cuda

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11

# Flags passed to nvcc compiler.
NVCCFLAGS += -g -arch=sm_30 --device-c --std=c++11 --expt-extended-lambda

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS_DEVICE = test_pf

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

all : test_device

clean :
	rm -f test_device gtest.a gtest_main.a *.o
	rm -rf *.dSYM

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Build tests

# pf.o : $(SRC_DIR)/pf.cu
# 	nvcc $(CPPFLAGS) $(NVCCFLAGS) -I$(TEST_DIR) -I$(SRC_DIR) -I$(INCLUDE_DIR) -I$(CUDA_DIR)/include $^ -c -o $@

# test_pf.o: $(TEST_DIR)/pf_test.cu
# 	nvcc $(CPPFLAGS) $(NVCCFLAGS) -I$(TEST_DIR) -I$(SRC_DIR) -I$(INCLUDE_DIR) -I$(CUDA_DIR)/include $^ -c -o $@

# test_device: pf.o test_pf.o gtest_main.a
# 	nvcc -arch=sm_30 $^ -lpthread $(LIB_FLAGS) -o $@

pf.o : $(SRC_DIR)/pf.cu $(INCLUDE_DIR)/pf/pf.cuh $(GTEST_HEADERS)
	nvcc $(CPPFLAGS) $(NVCCFLAGS) -I$(SRC_DIR) -I$(INCLUDE_DIR) -I$(CUDA_DIR)/include -c $(SRC_DIR)/pf.cu

pf_test.o : $(TEST_DIR)/pf_test.cu \
		$(INCLUDE_DIR)/pf/pf.cuh $(GTEST_HEADERS)
	nvcc $(CPPFLAGS) $(NVCCFLAGS) -I$(GTEST_DIR) -I$(SRC_DIR) -I$(INCLUDE_DIR) -I$(CUDA_DIR)/include -c $(TEST_DIR)/pf_test.cu

test_device : pf.o pf_test.o gtest_main.a
	nvcc -isystem $(GTEST_DIR)/include -arch=sm_30 -lpthread $^ -o $@

# pf.o : $(USER_DIR)/pf.cc $(USER_DIR)/pf.h $(GMOCK_HEADERS)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/pf.cc

# pf_test.o : $(TEST_DIR)/pf_test.cc \
# 		$(USER_DIR)/pf.h $(GMOCK_HEADERS)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(TEST_DIR)/pf_test.cc

# pf_test : pf.o pf_test.o gmock_main.a
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@


# demo.o : $(USER_DIR)/demo.cc $(USER_DIR)/demo.h
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/demo.cc

# demo : demo.o pf.o
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@
