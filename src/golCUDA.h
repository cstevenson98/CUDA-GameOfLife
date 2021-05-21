#include <curand_kernel.h>

__global__ void GolKernel_random(unsigned int* cellData, float density, int seed);
__global__ void GolKernel_next(unsigned int* cellData, unsigned int* cellNext);
