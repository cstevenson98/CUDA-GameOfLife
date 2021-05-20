#include "Renderer.h"
#include <iostream>

// void GLClearError()
// {
// 	while (glGetError() != GL_NO_ERROR);
// }

// /* ================================================== */

// bool GLLogCall(const char* function, const char* file, int line)
// {
// 	while (GLenum error = glGetError())
// 	{
// 		std::cout << "[OpelGL Error] (" << error << ")" << function <<
// 			" " << file << " : " << line << std::endl;
// 		return false;
// 	}
// 	return true;
// }

/* ================================================== */

void Renderer::Clear() const
{
	glClear(GL_COLOR_BUFFER_BIT);
}

void Renderer::Draw(const VertexArray& va, const Shader& shader, unsigned int count) const
{
	shader.Bind();
	// ib.Bind();
	va.Bind();
	// GLCall(glDrawElements(GL_TRIANGLES, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
	glDrawArrays(GL_POINTS, 0, count); // 3 indices starting at 0 -> 1 triangle
}
