GXX ?= g++
CUDA_PATH ?= /usr/local/cuda

all: test

test: conv conv_cl.s
	./conv input.png output.png conv_cl.s

conv_cl.s: conv.cl cl_stub.h
	clang -Xclang -finclude-default-header -x cl -cl-std=CL1.2 -S -target -nvptx64-nvidia-cuda -includecl_stub.h -O3 -o $@ $<

conv: conv.cpp
	$(GXX) -std=c++2a -Wall -Wextra -pedantic-errors -I./include -I$(CUDA_PATH)/include -lcuda -L$(CUDA_PATH)/lib64/stubs -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -lrt -lpthread -ldl -O3 -o $@ $^

clean:
	rm -f conv output.png conv_cl.s

.PHONY: all test clean
