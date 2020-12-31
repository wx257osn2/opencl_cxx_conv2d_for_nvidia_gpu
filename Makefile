GXX ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc

all: test

test: conv conv_cu.s
	./conv input.png output.png conv_cu.s

conv_cu.s: conv.cu
	$(NVCC) --ptx -O3 -c -o $@ $<

conv: conv.cpp
	$(GXX) -std=c++2a -Wall -Wextra -pedantic-errors -I./include -I$(CUDA_PATH)/include -lcuda -L$(CUDA_PATH)/lib64/stubs -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -lrt -lpthread -ldl -O3 -o $@ $^

clean:
	rm -f conv output.png conv_cu.s

.PHONY: all test clean
