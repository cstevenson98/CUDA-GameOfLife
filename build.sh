#!/bin/bash

CC=nvcc
LDFLAGS=`pkg-config --libs glew`
LDFLAGS="$LDFLAGS -lglut"
DEBUG=-lineinfo
WARN=
cd src/
$CC Shader.cpp $LDFLAGS $DEBUG -c
$CC golCUDA.cu $LDFLAGS $DEBUG -c
$CC golPipeline.cu $LDFLAGS $DEBUG -c
$CC main.cpp $LDFLAGS $DEBUG -c

$CC Shader.o golCUDA.o golPipeline.o main.o $LDFLAGS $DEBUG -o main
mv main ../main
cd ..