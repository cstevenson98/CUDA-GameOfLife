#pragma once
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <cuda.h>
#include <curand_kernel.h>

#include "VertexBuffer.h"
#include "VertexBufferLayout.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Renderer.h"

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

class GolCudaBuffer
{
public:
	GolCudaBuffer(dim3 threads, dim3 blocks, unsigned int fullCellWidth);

	~GolCudaBuffer();

	void load(std::string& filename, int xOffset, int yOffset);
	void random(float density, int seed);
	void next(); 
	void draw(Renderer renderer, unsigned int fullCellWidth);


	int m_genSkip;

private:
	struct cudaGraphicsResource* m_resource;
	size_t m_BufferSize;
	VertexBuffer m_vb;
	VertexArray m_va;
	VertexBufferLayout m_layout;
	Shader m_shader;
	unsigned int m_PointSize;
	dim3 threadSize;
	dim3 blockSize;
};

GolCudaBuffer::GolCudaBuffer(dim3 threads, dim3 blocks, unsigned int fullCellWidth)
	: threadSize ( threads ),
	  blockSize	 ( blocks ),
	  m_vb		 ( VertexBuffer(fullCellWidth*fullCellWidth*sizeof(unsigned int)) ),
	  m_shader	 ( Shader("res/shaders/GameOfLife.shader") ),
	  m_PointSize( 4 )
{

	glPointSize(m_PointSize);
	// Initialise the cuda parameters
	// ??
	// Initialise all of the GL objects
	m_BufferSize = fullCellWidth * fullCellWidth * sizeof(unsigned int);

	// Push layout just 1 u_int per site
	m_layout = VertexBufferLayout();
	m_layout.Push<unsigned int>(1); // state
	m_va.AddBuffer(m_vb, m_layout);

	// The shader here deals with the positions and has CPU-
	// adjustable width and colours and point size.
	
	m_shader.Bind();

	m_shader.SetUniformUint("fullCellWidth", fullCellWidth);
	m_shader.SetUniform4f("u_OnColour", 1., 1., 1., 1.);
	m_shader.SetUniform4f("u_OffColour", 0., 0., 0., 1.);
	m_shader.SetUniform4f("windowXY", -1.0, 1.0, -1.0, 1.0);
	// Bind the gl resource

	cudaGraphicsGLRegisterBuffer(&m_resource, m_vb.m_RendererID, cudaGraphicsRegisterFlagsNone);
}

GolCudaBuffer::~GolCudaBuffer()
{                                  
}


void GolCudaBuffer::next()
{
	// !! These pointers both have to be local to this method call.
	// For whatever reason, when you leave this method, it doesn't seem
	// like you can use m_DevState in the next iteration
	unsigned int* m_DevState;
	unsigned int* m_DevNextState;
	cudaMalloc((void**)&m_DevNextState, m_BufferSize);

	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);

	GolKernel_next<<<blockSize, threadSize>>>(m_DevState, m_DevNextState);
	cudaDeviceSynchronize();
	cudaMemcpy(m_DevState, m_DevNextState, m_BufferSize, cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &m_resource, 0);
	cudaFree(m_DevNextState);
}


void GolCudaBuffer::random(float density, int seed)
{
	unsigned int* m_DevState;
	unsigned int* m_DevNextState;
	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);
	GolKernel_random<<<blockSize, threadSize>>>(m_DevState, density, seed);
	cudaGraphicsUnmapResources(1, &m_resource, 0);
}

void GolCudaBuffer::draw(Renderer renderer, unsigned int fullCellWidth)
{
	renderer.Draw(m_va, m_shader, fullCellWidth * fullCellWidth);
}
