// Code taken from "TheCherno" YouTube series on OpenGL
// https://www.youtube.com/watch?v=W3gAzLwfIP0&list=PLlrATfBNZ98foTJPJ_Ev03o2oq3-GGOS2

#pragma once

class VertexBuffer
{
private:
	// ID allocated on VRAM
public:
	unsigned int m_RendererID;
	VertexBuffer(unsigned int size);
	~VertexBuffer();

	void Bind() const;
	void Unbind() const;
};