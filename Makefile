GXX ?= g++
CUDA_PATH ?= /usr/local/cuda

all: test

test: conv conv_clcpp.s
	./conv input.png output.png conv_clcpp.s

conv_clcpp.s: conv.clcpp cl_stub.hpp
	clang -Xclang -finclude-default-header -x cl -cl-std=CLC++ -S -target -nvptx64-nvidia-cuda -includecl_stub.hpp -O3 -o $@ $<

conv: conv.cpp
	$(GXX) -std=c++2a -Wall -Wextra -pedantic-errors -I./include -I$(CUDA_PATH)/include -lcuda -L$(CUDA_PATH)/lib64/stubs -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -lrt -lpthread -ldl -O3 -o $@ $^

clean:
	rm -f conv output.png conv_clcpp.s

.PHONY: all test clean
