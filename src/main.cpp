#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>

#include "lbPipeline.h"

const char* WindowTitle = "Conway's Game of Life";

///////////////////////////////////////////////////
///////////////////////////////////////////////////
const unsigned int pointSize = 2;

const unsigned int threadsPerBlockX = 20;
const unsigned int blockCountX = 64;

const unsigned int threadsPerBlockY = 20;
const unsigned int blockCountY = 36;

const unsigned int fullCellWidthX = threadsPerBlockX * blockCountX;
const unsigned int fullCellWidthY = threadsPerBlockY * blockCountY;

size_t GOLBufferSizeUint = fullCellWidthX * fullCellWidthY * sizeof(unsigned int);

dim3 threadSize(threadsPerBlockX, threadsPerBlockY);
dim3 blockSize(blockCountX, blockCountY);

CustomGraphicsPipeline Scene(threadSize, blockSize, fullCellWidthX, fullCellWidthY, pointSize);
///////////////////////////////////////////////////
///////////////////////////////////////////////////

static void RenderSceneCB()
{
        Scene.Draw();
}

int main(int argc, char** argv)
{
    // GLUT initialisation //
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);

    // Window Settings //
    glutInitWindowSize(fullCellWidthX * pointSize, fullCellWidthY * pointSize);
    glutCreateWindow(WindowTitle);
    glutFullScreen();

    GLenum res = glewInit();
    if (res != GLEW_OK) 
    {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return 1;
    }

    // OpenGL Initialisation //
    assert( Scene.Init() );

    // Glut config //
    glutDisplayFunc(RenderSceneCB);

    // Main loops //
    glutMainLoop();

    return 0;
}