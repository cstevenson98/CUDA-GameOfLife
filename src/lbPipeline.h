#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>

#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include "golCUDA.h"
#include "vectors.h"
#include "openGLutils.h" 
//#include "Shader.h"

class CustomGraphicsPipeline
{
   //private state variables
   public:
      GLuint m_VBO;
      Shader m_shader;
      dim3 m_threads;
      dim3 m_blocks;
      unsigned int m_fullCellWidthX;
      unsigned int m_fullCellWidthY;
      unsigned int m_pointSize;
      size_t m_BufferSize;
      struct cudaGraphicsResource* m_resource;
      
      // Constructor
      CustomGraphicsPipeline(dim3 threads, dim3 blocks, 
                              unsigned int fullCellWidthX, unsigned int fullCellWidthY, 
                              unsigned int pointSize) 
         :  m_threads         ( threads ),
            m_blocks          ( blocks ),
            m_fullCellWidthX  ( fullCellWidthX ),
            m_fullCellWidthY  ( fullCellWidthY ),
            m_pointSize       ( pointSize )
      {}

      // Destructor
      ~CustomGraphicsPipeline() 
      {
         glDisableVertexAttribArray(0);
      }

      bool Init();
      void Draw();
      
};