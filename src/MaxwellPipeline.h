//
// Created by conor on 26/10/24.
//

#ifndef MAXWELLPIPELINE_H
#define MAXWELLPIPELINE_H

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "openGLutils.h"

class MaxwellPipeline {
public:
    VertexBuffer m_vbField;
    VertexBuffer m_vbCharges;
    VertexArray m_vaField;
    VertexArray m_vaCharges;
    VertexBufferLayout m_layoutField;
    VertexBufferLayout m_layoutCharges;
    Shader m_shaderField;
    Shader m_shaderCharges;

    GLuint m_VBOField;
    GLuint m_VBOCharges;
    dim3 m_threads;
    dim3 m_blocks;
    unsigned int m_widthX;
    unsigned int m_widthY;
    size_t m_BufferSizeField;
    size_t m_BufferSizeCharges;
    struct cudaGraphicsResource *m_resourceField;
    struct cudaGraphicsResource *m_resourceCharges;

    MaxwellPipeline(dim3 threads, dim3 blocks, unsigned int widthX, unsigned int widthY);
    ~MaxwellPipeline();

    bool Init();
    void DrawField();
    void Draw();
    void UpdateField();
    void UpdateCharges();
    void Reset();
};


#endif //MAXWELLPIPELINE_H
