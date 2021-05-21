#include "golPipeline.h"

bool CustomGraphicsPipeline::Init()
{
    GLint ret = true;  
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 1.0f, Alpha = 0.0f;

    glClearColor(Red, Green, Blue, Alpha);

    return ret;
}

void CustomGraphicsPipeline::Draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers();
}