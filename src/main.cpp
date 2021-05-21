#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>

#include "golPipeline.h"

const char* WindowTitle = "Conway's Game of Life";

///////////////////////////////////////////////////
CustomGraphicsPipeline Scene;
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
    int width = 500; int height = 500;
    int x = 200; int y = 100;
    glutInitWindowSize(width, height);
    glutInitWindowPosition(x, y);
    glutCreateWindow(WindowTitle);

    // OpenGL Initialisation //
    assert( Scene.Init() );

    // Glut config //
    glutDisplayFunc(RenderSceneCB);

    // Main loops //
    glutMainLoop();

    return 0;
}