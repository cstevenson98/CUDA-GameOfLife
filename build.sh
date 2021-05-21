#!/bin/bash

CC=nvcc
LDFLAGS=`pkg-config --libs glew`
LDFLAGS="$LDFLAGS -lglut"


cd src/
$CC golCUDA.cu $LDFLAGS -c
$CC golPipeline.cpp $LDFLAGS -c
$CC main.cpp $LDFLAGS -c

$CC golPipeline.o main.o $LDFLAGS -o main
mv main ../main
cd ..