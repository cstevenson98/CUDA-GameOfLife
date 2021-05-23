#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>

#include "golPipeline.h"

const char* WindowTitle = "Conway's Game of Life";

///////////////////////////////////////////////////
const unsigned int pointSize = 4;

const unsigned int threadsPerBlock = 18;
const unsigned int blockCount = 4;

const unsigned int fullCellWidth = threadsPerBlock * blockCount;
size_t GOLBufferSizeUint = fullCellWidth * fullCellWidth * sizeof(unsigned int);

dim3 threadSize(threadsPerBlock, threadsPerBlock);
dim3 blockSize(blockCount, blockCount);

CustomGraphicsPipeline Scene(threadSize, blockSize, fullCellWidth, pointSize);
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
    glutInitWindowSize(fullCellWidth * (pointSize+1), fullCellWidth * (pointSize+1));
    glutCreateWindow(WindowTitle);

    GLenum res = glewInit();
    if (res != GLEW_OK) {
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
