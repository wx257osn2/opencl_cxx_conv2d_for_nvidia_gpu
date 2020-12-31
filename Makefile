GXX ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc

all: test

test: conv
	./conv input.png output.png

conv_cu.o: conv.cu
	$(NVCC) -O3 -c -o $@ $<

conv.o: conv.cpp
	$(GXX) -std=c++2a -Wall -Wextra -pedantic-errors -I./include -I$(CUDA_PATH)/include -O3 -c -o $@ $<

conv: conv.o conv_cu.o
	$(GXX) -lcuda -L$(CUDA_PATH)/lib64/stubs -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lcudart -O3 -o $@ $^

clean:
	rm -f conv output.png conv_cu.o conv.o

.PHONY: all test clean
