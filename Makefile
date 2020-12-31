GXX ?= g++

all: test

test: conv
	./conv input.png output.png

conv: conv.cpp
	$(GXX) -std=c++2a -Wall -Wextra -pedantic-errors -I./include -O3 -o $@ $<

clean:
	rm -f conv output.png

.PHONY: all test clean
