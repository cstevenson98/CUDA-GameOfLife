#include "golCUDA.h"

__global__ void GolKernel_random(unsigned int* cellData, float density, int seed)
{
	unsigned int xId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int yId = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int idx = gridDim.x * blockDim.x * yId + xId;

	curandState localState;
	curand_init(seed, idx, 0, &localState);

	cellData[idx] = (curand_uniform(&localState) > density ? 1 : 0);
}

__global__ void GolKernel_next(unsigned int* cellData, unsigned int* cellNext)
{
	int xId = threadIdx.x + blockIdx.x * blockDim.x;
	int yId = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = gridDim.x * blockDim.x * yId + xId;

	unsigned int state = cellData[idx];
	int nextState;

	if ((xId < (gridDim.x * blockDim.x)) && (xId > 0) && (yId > 0) && (yId < (gridDim.y * blockDim.y)))
	{
		unsigned int sum = 0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				idx = gridDim.x * blockDim.x * (yId + j - 1) + (xId + i - 1);
				sum += cellData[idx];
				//std::printf("%d", sum);
			}
		}
		sum -= state;
		nextState = ( sum==3 || (state==1 && sum==2) );
	}
	else {
		nextState = 0;
	}

	idx = gridDim.x * blockDim.x * yId + xId;
	cellNext[idx] = nextState;

}

__global__ void GolKernel_copy(unsigned int* cellData, unsigned int* cellNext)
{
	int xId = threadIdx.x + blockIdx.x * blockDim.x;
	int yId = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = gridDim.x * blockDim.x * yId + xId;

	cellData[idx] = cellNext[idx];
}
