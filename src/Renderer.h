#pragma once

#include <GL/glew.h>
#include <iostream> 

#include "VertexBuffer.h"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

// #define ASSERT(x) if(!(x)) __debugbreak(); 
// // Note we don't need the semi-colon at the end
// // as that will be placed after calling this macro
// #define GLCall(x) GLClearError();\
// 	x;\
// 	ASSERT(GLLogCall(#x, __FILE__, __LINE__))

// /* ================================================== */

// void GLClearError();
// bool GLLogCall(const char* function, const char* file, int line);

/* ================================================== */

class Renderer 
{
public:
	void Clear() const;
	void Draw(const VertexArray& va, const Shader& shader, unsigned int count) const;
};