#include "gol/Utility.h"

#include "gol/golPipeline.hu"
#include <iostream>

bool GoLPipeline::Init() {
  // Check CUDA device properties at initialization
  checkCudaDevice();

  // Check if CUDA-GL interop is supported
  int deviceCount;
  gpuErrchk(cudaGetDeviceCount(&deviceCount));

  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));

  if (!deviceProp.canMapHostMemory) {
    fprintf(stderr, "Device does not support mapping host memory\n");
    return false;
  }

  // Get current device
  int currentDevice;
  gpuErrchk(cudaGetDevice(&currentDevice));

  // Only set device if it's not already set
  if (currentDevice != 0) {
    gpuErrchk(cudaSetDevice(0));
  }

  // Initialize GL context first
  GLint ret = true;
  GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;

  glClearColor(Red, Green, Blue, Alpha);
  glPointSize(m_pointSize);

  // Init Buffer
  glGenBuffers(1, &m_VBO);
  glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
  glBufferData(GL_ARRAY_BUFFER, m_widthX * m_widthY * sizeof(unsigned int), 0,
               GL_DYNAMIC_DRAW);

  // Attrib Pointer
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

  // ShaderS
  m_shader =
      Shader("/home/conor/dev/CUDA-GameOfLife/shaders/GameOfLife.shader");

  // Now initialize CUDA-GL interop
  cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_resource, m_VBO,
                                                 cudaGraphicsRegisterFlagsNone);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to register buffer: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Trying alternative registration method...\n");

    // Try alternative registration method
    err = cudaGraphicsGLRegisterBuffer(&m_resource, m_VBO,
                                       cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
      fprintf(stderr, "Alternative registration also failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }
  }

  unsigned int *m_DevState;
  m_BufferSize = m_widthX * m_widthY * sizeof(unsigned int);

  err = cudaGraphicsMapResources(1, &m_resource, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to map resources: %s\n", cudaGetErrorString(err));
    return false;
  }

  err = cudaGraphicsResourceGetMappedPointer((void **)&m_DevState,
                                             &m_BufferSize, m_resource);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to get mapped pointer: %s\n",
            cudaGetErrorString(err));
    cudaGraphicsUnmapResources(1, &m_resource, 0);
    return false;
  }

  GolKernel_random<<<m_blocks, m_threads>>>(m_DevState, 0.75f, 0);
  gpuKernelErrchk();
  gpuSyncErrchk();

  err = cudaGraphicsUnmapResources(1, &m_resource, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to unmap resources: %s\n", cudaGetErrorString(err));
    return false;
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return ret;
}

void GoLPipeline::Draw() {
  m_shader.Bind();
  m_shader.SetUniformUint("widthX", m_widthX);
  m_shader.SetUniformUint("widthY", m_widthY);
  m_shader.SetUniform4f("u_OnColour", 1., 1., 1., 1.);
  m_shader.SetUniform4f("u_OffColour", 0., 0., 0., 1.);
  m_shader.SetUniform4f("windowXY", -1.0, 1.0, -1.0, 1.0);

  glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
  glDrawArrays(GL_POINTS, 0, m_widthX * m_widthY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  m_shader.Unbind();
}

void GoLPipeline::Update(unsigned int generations) {
  unsigned int *m_DevState;
  unsigned int *m_DevNextState;
  gpuErrchk(cudaMalloc((void **)&m_DevNextState, m_BufferSize));

  gpuErrchk(cudaGraphicsMapResources(1, &m_resource, 0));
  gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&m_DevState,
                                                 &m_BufferSize, m_resource));

  for (unsigned int i = 0; i < generations; i++) {
    GolKernel_next<<<m_blocks, m_threads>>>(m_DevState, m_DevNextState);
    gpuKernelErrchk();
    gpuSyncErrchk();

    gpuErrchk(cudaMemcpy(m_DevState, m_DevNextState, m_BufferSize,
                         cudaMemcpyDeviceToDevice));
  }

  gpuErrchk(cudaGraphicsUnmapResources(1, &m_resource, 0));
  gpuErrchk(cudaFree(m_DevNextState));
}

void GoLPipeline::Reset() {
  unsigned int *m_DevState;
  gpuErrchk(cudaGraphicsMapResources(1, &m_resource, 0));
  gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **)&m_DevState,
                                                 &m_BufferSize, m_resource));
  GolKernel_random<<<m_blocks, m_threads>>>(m_DevState, 0.75f, 0);
  gpuKernelErrchk();
  gpuSyncErrchk();

  gpuErrchk(cudaGraphicsUnmapResources(1, &m_resource, 0));
}