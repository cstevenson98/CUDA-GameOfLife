#include "openGLutils.h"

//////////////////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING 
//////////////////////////////////////////////////////////////////////////////////////////

void GLClearError()
{
	while (glGetError() != GL_NO_ERROR);
}

bool GLLogCall(const char* function, const char* file, int line)
{
	while (GLenum error = glGetError())
	{
		std::cout << "[OpelGL Error] (" << error << ")" << function <<
			" " << file << " : " << line << std::endl;
		return false;
	}
	return true;
}


//////////////////////////////////////////////////////////////////////////////////////////
// VERTEX BUFFERS
//////////////////////////////////////////////////////////////////////////////////////////

VertexBuffer::VertexBuffer(unsigned int size)
{
	GLCall(glGenBuffers(1, &m_RendererID));
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
	GLCall(glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW));
}

VertexBuffer::~VertexBuffer()
{
	GLCall(glDeleteBuffers(1, &m_RendererID))
}

void VertexBuffer::Bind() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
}

void VertexBuffer::Unbind() const
{
	GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}


//////////////////////////////////////////////////////////////////////////////////////////
// INDEX ARRAYS
//////////////////////////////////////////////////////////////////////////////////////////

IndexBuffer::IndexBuffer(const unsigned int* data, unsigned int count)
	: m_Count(count)
{
	ASSERT(sizeof(unsigned int) == sizeof(GLuint));

	GLCall(glGenBuffers(1, &m_RendererID));
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * count, data, GL_STATIC_DRAW));
}

IndexBuffer::~IndexBuffer()
{
	GLCall(glDeleteBuffers(1, &m_RendererID))
}


void IndexBuffer::Bind() const
{
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
}

void IndexBuffer::Unbind() const
{
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}


//////////////////////////////////////////////////////////////////////////////////////////
// VERTEX ARRAYS
//////////////////////////////////////////////////////////////////////////////////////////

VertexArray::VertexArray()
{
	GLCall(glGenVertexArrays(1, &m_RendererID));
}

VertexArray::~VertexArray()
{
	GLCall(glDeleteVertexArrays(1, &m_RendererID));
}

void VertexArray::AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout)
{
	Bind();
	vb.Bind();
	const auto& elements = layout.GetElements();
	unsigned int offset = 0;
	for (unsigned int i = 0; i < elements.size(); i++)
	{
		const auto& element = elements[i];
		GLCall(glEnableVertexAttribArray(i));
		GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalised, 
										layout.GetStride(), /*(const void*)*/&offset));
		offset += element.count * VertexBufferElement::GetSizeOfType(element.type);
	}
}

void VertexArray::Bind() const
{
	GLCall(glBindVertexArray(m_RendererID));
}

void VertexArray::Unbind() const
{
	GLCall(glBindVertexArray(0));
}


//////////////////////////////////////////////////////////////////////////////////////////
// SHADERS
//////////////////////////////////////////////////////////////////////////////////////////

Shader::Shader(const std::string& filepath)
	: m_FilePath(filepath), m_RendererID(0)
{
	ShaderProgramSource source = ParseShader(filepath);

	std::cout << "VERTEX" << std::endl << source.VertexSource << std::endl;
	std::cout << "FRAGMENT" << std::endl << source.FragmentSource << std::endl;

	m_RendererID = CreateShader(source.VertexSource, source.FragmentSource);

	GLCall( glUseProgram(m_RendererID) );
}

Shader::~Shader()
{
	GLCall( glDeleteProgram(m_RendererID) );
}

void Shader::Bind() const
{
	GLCall( glUseProgram(m_RendererID) );
}

void Shader::Unbind() const
{
	GLCall( glUseProgram(0) );
}

int Shader::GetUniformLocation(const std::string& name)
{
	if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
		return m_UniformLocationCache[name];

	GLCall( int location = glGetUniformLocation(m_RendererID, name.c_str()) );
	if (location == -1)
		std::cout << "No active uniform variable with name " << name << " found" << std::endl;

	m_UniformLocationCache[name] = location;

	return location;
}

void Shader::SetUniform1f(const std::string& name, float value)
{
	GLCall( glUniform1f(GetUniformLocation(name), value) );
}

void Shader::SetUniform1fv(const std::string& name, std::vector<float> data)
{
	GLCall( glUniform1fv(GetUniformLocation(name), data.size(), data.data()));
}

void Shader::SetUniformUint(const std::string& name, unsigned int value)
{
	GLCall( glUniform1ui(GetUniformLocation(name), value) );
}

void Shader::SetUniform4f(const std::string& name, float f0, float f1, float f2, float f3)
{
	GLCall( glUniform4f(GetUniformLocation(name), f0, f1, f2, f3) );
}

enum ShaderType
{
	NONE = -1, VERTEX = 0, FRAGMENT = 1
};

struct ShaderProgramSource Shader::ParseShader(const std::string& filepath)
{

	std::ifstream stream(filepath);
	std::string line;
	std::stringstream ss[2];
	ShaderType type = NONE;

	while (getline(stream, line))
	{
		if (line.find("#shader") != std::string::npos)
		{
			if (line.find("vertex") != std::string::npos)
				type = VERTEX;
			else if (line.find("fragment") != std::string::npos)
				type = FRAGMENT;
		}
		else
		{
			ss[(int)type] << line << '\n';
		}
	}

	struct ShaderProgramSource sps = { ss[0].str(), ss[1].str() };
	return sps;
}

unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
{
	GLCall( unsigned int id = glCreateShader(type) );
	const char* src = source.c_str();
	GLCall( glShaderSource(id, 1, &src, nullptr) );
	GLCall( glCompileShader(id) );

	// Error handling
	int result;
	GLCall( glGetShaderiv(id, GL_COMPILE_STATUS, &result) );
	std::cout << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment") << " shader compile status: " << result << std::endl;
	if ( result == GL_FALSE )
	{
		int length;
		GLCall( glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length) );
		char* message = (char*) alloca(length * sizeof(char));
		GLCall( glGetShaderInfoLog(id, length, &length, message) );
		std::cout 
			<< "Failed to compile "
			<< (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
			<< "shader"
			<< std::endl;
		std::cout << message << std::endl;
		GLCall( glDeleteShader(id) );
		return 0;
	}

	return id;
}

unsigned int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
	// create a shader program
	unsigned int program = glCreateProgram();
	unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

	GLCall( glAttachShader(program, vs) );
	GLCall( glAttachShader(program, fs) );

	GLCall( glLinkProgram(program) );

	GLint program_linked;

	GLCall( glGetProgramiv(program, GL_LINK_STATUS, &program_linked) );
	std::cout << "Program link status: " << program_linked << std::endl;
	if (program_linked != GL_TRUE)
	{
		GLsizei log_length = 0;
		GLchar message[1024];
		GLCall( glGetProgramInfoLog(program, 1024, &log_length, message) );
		std::cout << "Failed to link program" << std::endl;
		std::cout << message << std::endl;
	}

	GLCall( glValidateProgram(program) );

	GLCall( glDeleteShader(vs) );
	GLCall( glDeleteShader(fs) );

	return program;
}

//////////////////////////////////////////////////////////////////////////////////////////
// RENDERER
//////////////////////////////////////////////////////////////////////////////////////////

void Renderer::Clear() const
{
	GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

void Renderer::Draw(const VertexArray& va, const Shader& shader, unsigned int count) const
{
	shader.Bind();
	// ib.Bind();
	va.Bind();
	// GLCall(glDrawElements(GL_TRIANGLES, ib.GetCount(), GL_UNSIGNED_INT, nullptr));
	glDrawArrays(GL_POINTS, 0, count); // 3 indices starting at 0 -> 1 triangle
}
