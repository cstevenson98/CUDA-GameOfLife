#!/bin/bash

CC=nvcc
LDFLAGS=-lGLEW -lGLU -lGL -lglfw
DEBUG=-lineinfo

OBJ=src/main.o src/golPipeline.o src/golCUDA.o src/openGLutils.o
all: $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) $(DEBUG) -o main

%.o: %.cpp
	$(CC) $(LDFLAGS) $(DEBUG) -c $< -o $@

%.o: %.cu
	$(CC) $(LDFLAGS) $(DEBUG) -c $< -o $@

clean:
	rm -v src/*.o
