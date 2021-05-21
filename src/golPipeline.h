#pragma once

#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>

#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include "golCUDA.h"

class CustomGraphicsPipeline
{
   //private state variables
   GLuint variable_x;
   public:
      CustomGraphicsPipeline() : variable_x(1) { }
      ~CustomGraphicsPipeline() {}

      bool Init();
      void Draw();
};