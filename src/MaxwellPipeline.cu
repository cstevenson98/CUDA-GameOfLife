//
// Created by conor on 26/10/24.
//

#include "MaxwellPipeline.h"
#include <iostream>

MaxwellPipeline::MaxwellPipeline(dim3 threads, dim3 blocks, unsigned int widthX, unsigned int widthY)
    : m_threads(threads), m_blocks(blocks), m_widthX(widthX), m_widthY(widthY) {
}

MaxwellPipeline::~MaxwellPipeline() {
    glDisableVertexAttribArray(0);
}

constexpr int Dimensions = 2;
constexpr int FieldLinesPerCharge = 18;
constexpr int FieldLinesBufferSize = 1000;
constexpr int ChargeNumber = 2;

bool MaxwellPipeline::Init() {
    GLint ret = true;
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;

    glClearColor(Red, Green, Blue, Alpha);

    // charges 2D
    float chargePos[] = {
        -0.333f, 0.0f,
        0.333f, 0.0f
    };

    // For the number of pixel, initialise a field which is simply 0.0 everywhere
    float fieldStrength[Dimensions * m_widthX * m_widthY];
    for (int i = 0; i < Dimensions * m_widthX * m_widthY; i++) {
        fieldStrength[i] = 0.0f;
    }

    float fieldLines[ChargeNumber * Dimensions * FieldLinesPerCharge * FieldLinesBufferSize];
    for (int i = 0; i < ChargeNumber * Dimensions * FieldLinesPerCharge * FieldLinesBufferSize; i++) {
        fieldLines[i] = 0.0f;
    }

    m_BufferSizeField = Dimensions * m_widthX * m_widthY * sizeof(float);
    m_BufferSizeFieldLines = ChargeNumber * Dimensions * sizeof(float);
    m_BufferSizeCharges = ChargeNumber * Dimensions * sizeof(float);

    // Init Field Buffer
    glGenBuffers(1, &m_VBOField);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOField);
    glBufferData(GL_ARRAY_BUFFER, Dimensions * m_widthX * m_widthY * sizeof(float), fieldStrength, GL_DYNAMIC_DRAW);

    // Init Field Lines Buffer
    glGenBuffers(1, &m_VBOFieldLines);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOFieldLines);
    glBufferData(
        GL_ARRAY_BUFFER, ChargeNumber * Dimensions * FieldLinesPerCharge * FieldLinesBufferSize * sizeof(float),
        fieldLines, GL_DYNAMIC_DRAW);

    // Init Charges Buffer
    glGenBuffers(1, &m_VBOCharges);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOCharges);
    glBufferData(GL_ARRAY_BUFFER, Dimensions * 2 * sizeof(float), chargePos, GL_DYNAMIC_DRAW);

    // Shaders
    m_shaderField = Shader("/home/conor/dev/CUDA-GameOfLife/shaders/Field.shader");
    m_shaderCharges = Shader("/home/conor/dev/CUDA-GameOfLife/shaders/Charges.shader");

    // CUDA graphics resource
    cudaGraphicsGLRegisterBuffer(&m_resourceField, m_VBOField, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&m_resourceCharges, m_VBOCharges, cudaGraphicsRegisterFlagsNone);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return ret;
}

void MaxwellPipeline::Draw() {
    m_shaderField.Bind();
    // set uniforms
    m_shaderField.SetUniformUint("xWidth", m_widthX);
    m_shaderField.SetUniformUint("yWidth", m_widthY);
    // Field
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOField);
    // Attrib
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) 0);

    glDrawArrays(GL_POINTS, 0, m_widthX * m_widthY);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    m_shaderField.Unbind();
    m_shaderCharges.Bind();
    // bind buffers
    glBindBuffer(GL_ARRAY_BUFFER, m_VBOCharges);
    // Attrib
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) 0);

    // Draw
    glDrawArrays(GL_POINTS, 0, 2);

    // unbnd
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    m_shaderCharges.Unbind();
}

// Update field kernel
__global__ void UpdateFieldKernel(float *field, const float *charges, unsigned int chargeCount, unsigned int widthX,
                                  unsigned int widthY) {
    int xId = threadIdx.x + blockIdx.x * blockDim.x;
    int yId = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = yId * widthX + xId;

    if (xId >= widthX || yId >= widthY) return;

    // positions on a scale of -1 to 1
    float x = (2.0f * xId / (widthX - 1)) - 1.0f;
    float y = (2.0f * yId / (widthY - 1)) - 1.0f;

    float fieldMagnitude = 0.0f;
    field[2 * idx] = 0.;
    field[2 * idx + 1] = 0.;
    // field sum
    for (unsigned int i = 0; i < chargeCount; i++) {
        float chargeX = charges[2 * i];
        float chargeY = charges[2 * i + 1];

        float dx = chargeX - x;
        float dy = chargeY - y;
        float distanceSquared = dx * dx + dy * dy;
        float distance = sqrt(distanceSquared);
        float fieldStrength = 1.0f / distanceSquared; // Coulomb's law (assuming unit charge)

        field[2 * idx] += fieldStrength * dx / distance;
        field[2 * idx + 1] += fieldStrength * dy / distance;
    }
}

void MaxwellPipeline::UpdateField() {
    float *m_DevField;
    cudaGraphicsMapResources(1, &m_resourceField, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevField, &m_BufferSizeField, m_resourceField);

    float *m_DevCharges;
    cudaGraphicsMapResources(1, &m_resourceCharges, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevCharges, &m_BufferSizeCharges, m_resourceCharges);

    // Call CUDA kernel to update field
    UpdateFieldKernel<<<m_blocks, m_threads>>>(m_DevField, m_DevCharges, 2, m_widthX, m_widthY);

    cudaGraphicsUnmapResources(1, &m_resourceField, 0);
    cudaGraphicsUnmapResources(1, &m_resourceCharges, 0);
}

__global__ void UpdateFieldLinesKernel(const float *field, float *fieldLines, const float *charges,
                                       unsigned int chargeCount, unsigned int fieldLinesBufferSize) {
    // For each charge, calculate the field lines by following the field gradient
    // Each field line in [0, fieldLinesPerCharge) starts at a different angle around the charge

    int chargeId = blockIdx.x;
    int fieldLineId = threadIdx.x;

    if (chargeId >= chargeCount) return;
}

void MaxwellPipeline::UpdateFieldLines() {
    float *m_DevField;
    cudaGraphicsMapResources(1, &m_resourceField, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevField, &m_BufferSizeField, m_resourceField);

    float *m_DevFieldLines;
    cudaGraphicsMapResources(1, &m_resourceFieldLines, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevFieldLines, &m_BufferSizeFieldLines, m_resourceFieldLines);

    float *m_DevCharges;
    cudaGraphicsMapResources(1, &m_resourceCharges, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevCharges, &m_BufferSizeCharges, m_resourceCharges);

    // Call CUDA kernel to update field lines, in a 1D launch configuration
    UpdateFieldLinesKernel<<<ChargeNumber, FieldLinesPerCharge>>>(m_DevField, m_DevFieldLines, m_DevCharges,
                                                                  ChargeNumber, FieldLinesBufferSize);

    cudaGraphicsUnmapResources(1, &m_resourceFieldLines, 0);
}


void MaxwellPipeline::UpdateCharges() {
    float *m_DevCharges;
    cudaGraphicsMapResources(1, &m_resourceCharges, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevCharges, &m_BufferSizeCharges, m_resourceCharges);

    // Call CUDA kernel to update charges
    // Example: UpdateChargesKernel<<<m_blocks, m_threads>>>(m_DevCharges);

    cudaGraphicsUnmapResources(1, &m_resourceCharges, 0);
}

void MaxwellPipeline::Reset() {
    float *m_DevField;
    float *m_DevCharges;
    cudaGraphicsMapResources(1, &m_resourceField, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevField, &m_BufferSizeField, m_resourceField);
    cudaGraphicsMapResources(1, &m_resourceCharges, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &m_DevCharges, &m_BufferSizeCharges, m_resourceCharges);

    // Call CUDA kernel to reset field and charges
    // Example: ResetFieldKernel<<<m_blocks, m_threads>>>(m_DevField);
    // Example: ResetChargesKernel<<<m_blocks, m_threads>>>(m_DevCharges);

    cudaGraphicsUnmapResources(1, &m_resourceField, 0);
    cudaGraphicsUnmapResources(1, &m_resourceCharges, 0);
}
