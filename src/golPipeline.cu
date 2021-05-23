#include "golPipeline.h"
#include <iostream>

bool CustomGraphicsPipeline::Init()
{
    GLint ret = true;  
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 1.0f, Alpha = 0.0f;

    glClearColor(Red, Green, Blue, Alpha);
    glPointSize(m_pointSize);

    //Vector3f Vertices[1];
    //Vertices[0] = Vector3f(0.0f, 0.0f, 0.0f);

    // Init Buffer
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_fullCellWidth * m_fullCellWidth * sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);

    // Attrib Pointer
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);
   
    // Shader
    m_shader = Shader("shaders/GameOfLife.shader");
    m_shader.Bind();
	m_shader.SetUniformUint("fullCellWidth", m_fullCellWidth);
	m_shader.SetUniform4f("u_OnColour", 1., 1., 1., 1.);
	m_shader.SetUniform4f("u_OffColour", 0., 0., 0., 1.);
	m_shader.SetUniform4f("windowXY", -1.0, 1.0, -1.0, 1.0);

    // CUDA graphics resource 
    cudaGraphicsGLRegisterBuffer(&m_resource, m_VBO, cudaGraphicsRegisterFlagsNone);

    unsigned int* m_DevState;
    m_BufferSize = m_fullCellWidth * m_fullCellWidth * sizeof(unsigned int);
	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);
	GolKernel_random<<<m_blocks, m_threads>>>(m_DevState, 0.5f, 0);
	cudaGraphicsUnmapResources(1, &m_resource, 0);

    return ret;
}

void CustomGraphicsPipeline::Draw()
{

    std::cout << "draw!" << std::endl;
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    unsigned int* m_DevState;
	unsigned int* m_DevNextState;
	cudaMalloc((void**)&m_DevNextState, m_BufferSize);

	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);

	GolKernel_next<<<m_blocks, m_threads>>>(m_DevState, m_DevNextState);
	cudaDeviceSynchronize();
	cudaMemcpy(m_DevState, m_DevNextState, m_BufferSize, cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &m_resource, 0);
	cudaFree(m_DevNextState);

    glDrawArrays(GL_POINTS, 0, m_fullCellWidth * m_fullCellWidth);

    glutSwapBuffers();
    glutPostRedisplay();
}