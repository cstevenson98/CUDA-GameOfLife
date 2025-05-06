// C Stevenson 2021

#pragma once

/*

        FILE READING AND WRITING

*/

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Basic CUDA error checking macro
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

// CUDA kernel error checking macro
#define gpuKernelErrchk()                \
  {                                      \
    gpuKernelAssert(__FILE__, __LINE__); \
  }

// CUDA device synchronization error checking macro
#define gpuSyncErrchk()                \
  {                                    \
    gpuSyncAssert(__FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

inline void gpuKernelAssert(const char *file, int line, bool abort = true) {
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Kernel Error: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort) exit(code);
  }
}

inline void gpuSyncAssert(const char *file, int line, bool abort = true) {
  cudaError_t code = cudaDeviceSynchronize();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Sync Error: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort) exit(code);
  }
}

// Helper function to check CUDA device properties
inline void checkCudaDevice() {
  int deviceCount;
  gpuErrchk(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    exit(1);
  }

  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));

  printf("Using CUDA Device: %s\n", deviceProp.name);
  printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
  printf("Max Threads per MultiProcessor: %d\n",
         deviceProp.maxThreadsPerMultiProcessor);
  printf("Max Blocks per MultiProcessor: %d\n",
         deviceProp.maxBlocksPerMultiProcessor);
  printf("Max Shared Memory per Block: %zu bytes\n",
         deviceProp.sharedMemPerBlock);
  printf("Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
}