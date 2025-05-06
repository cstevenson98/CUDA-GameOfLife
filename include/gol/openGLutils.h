#pragma once
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

//////////////////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING 
//////////////////////////////////////////////////////////////////////////////////////////

#define ASSERT(x) if(!(x)) exit(-1); 

#define GLCall(x) GLClearError();\
	x;\
	ASSERT(GLLogCall(#x, __FILE__, __LINE__))


void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);


//////////////////////////////////////////////////////////////////////////////////////////
// VERTEX BUFFER LAYOUTS
//////////////////////////////////////////////////////////////////////////////////////////

struct VertexBufferElement
{
	unsigned int type;
	unsigned int count;
	unsigned char normalised;

	static unsigned int GetSizeOfType(unsigned int type)
	{
		switch (type)
		{
			case GL_FLOAT: return 4;
			case GL_UNSIGNED_INT: return 4;
			case GL_UNSIGNED_BYTE: return 1;
		}
		ASSERT(false);
		return 0;
	}	
};

class VertexBufferLayout
{
private:
	std::vector<VertexBufferElement> m_Elements;
	unsigned int m_Stride;
public:
	VertexBufferLayout()
		: m_Stride(0) {}

	void Push_float(unsigned int count)
	{
		m_Elements.push_back({ GL_FLOAT, count, GL_FALSE });
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_FLOAT);
	}

	void Push_uint(unsigned int count)
	{
		m_Elements.push_back({ GL_UNSIGNED_INT, count, GL_FALSE });
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_INT);
	}

	void Push_uchar(unsigned int count)
	{
		m_Elements.push_back({ GL_UNSIGNED_BYTE, count, GL_TRUE });
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_BYTE);
	}

	inline const std::vector<VertexBufferElement> GetElements() const { return m_Elements; }
	inline unsigned int GetStride() const { return m_Stride; }

};


//////////////////////////////////////////////////////////////////////////////////////////
// VERTEX BUFFERS
//////////////////////////////////////////////////////////////////////////////////////////

class VertexBuffer
{
private:
	// ID allocated on VRAM
public:
	unsigned int m_RendererID;
	VertexBuffer(){ };
	VertexBuffer(unsigned int size);
	~VertexBuffer();

	void Bind() const;
	void Unbind() const;
};


//////////////////////////////////////////////////////////////////////////////////////////
// INDEX BUFFERS
//////////////////////////////////////////////////////////////////////////////////////////

class IndexBuffer
{
private:
	// ID allocated on VRAM
	unsigned int m_RendererID;
	unsigned int m_Count;
public:
	// Might need to make unsigned long long
	IndexBuffer(const unsigned int* data, unsigned int count);
	~IndexBuffer();

	void Bind() const;
	void Unbind() const;

	inline unsigned int GetCount() const { return m_Count; }
};


//////////////////////////////////////////////////////////////////////////////////////////
// VERTEX BUFFERS
//////////////////////////////////////////////////////////////////////////////////////////

class VertexArray
{
private:
public:
	unsigned int m_RendererID;
	VertexArray();
	~VertexArray();

	void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout);

	void Bind() const;
	void Unbind() const;
};


//////////////////////////////////////////////////////////////////////////////////////////
// SHADERS
//////////////////////////////////////////////////////////////////////////////////////////

struct ShaderProgramSource
{
	std::string VertexSource;
	std::string FragmentSource;
};

class Shader
{
private:
	unsigned int m_RendererID;
	std::string m_FilePath;
	std::unordered_map<std::string, int> m_UniformLocationCache;

public:
	Shader() {};
	Shader(const std::string& filepath);
	~Shader();

	void Bind() const;
	void Unbind() const;

	// Set uniforms
	void SetUniformUint(const std::string& name, unsigned int value);
	void SetUniform4f(const std::string& name, float f0, float f1, float f2, float f3);
	void SetUniform1f(const std::string& name, float value);
	void SetUniform1fv(const std::string& name, std::vector<float> data);

private:
	int GetUniformLocation(const std::string& name);
	struct ShaderProgramSource ParseShader(const std::string& filepath);
	unsigned int CompileShader(unsigned int type, const std::string& source);
	unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);

};


//////////////////////////////////////////////////////////////////////////////////////////
// RENDERER
//////////////////////////////////////////////////////////////////////////////////////////

class Renderer 
{
public:
	void Clear() const;
	void Draw(const VertexArray& va, const Shader& shader, unsigned int count) const;
};