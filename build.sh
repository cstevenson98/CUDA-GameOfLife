#!/bin/bash

CC=nvcc
LDFLAGS=`pkg-config --libs glew`
LDFLAGS="$LDFLAGS -lglut"


cd src/
$CC Shader.cpp $LDFLAGS -c
$CC golCUDA.cu $LDFLAGS -c
$CC golPipeline.cu $LDFLAGS -c
$CC main.cpp $LDFLAGS -c

$CC Shader.o golCUDA.o golPipeline.o main.o $LDFLAGS -o main
mv main ../main
cd ..