// Code taken from "TheCherno" YouTube series on OpenGL
// https://www.youtube.com/watch?v=W3gAzLwfIP0&list=PLlrATfBNZ98foTJPJ_Ev03o2oq3-GGOS2

#include "Renderer.h"
#include "Shader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>



Shader::Shader(const std::string& filepath)
	: m_FilePath(filepath), m_RendererID(0)
{
	ShaderProgramSource source = ParseShader(filepath);

	//std::cout << "VERTEX" << std::endl << source.VertexSource << std::endl;
	//std::cout << "FRAGMENT" << std::endl << source.FragmentSource << std::endl;

	m_RendererID = CreateShader(source.VertexSource, source.FragmentSource);

	glUseProgram(m_RendererID);
}



Shader::~Shader()
{
	glDeleteProgram(m_RendererID);
}



void Shader::Bind() const
{
	glUseProgram(m_RendererID);
}



void Shader::Unbind() const
{
	glUseProgram(0);
}



int Shader::GetUniformLocation(const std::string& name)
{
	if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
		return m_UniformLocationCache[name];

	int location = glGetUniformLocation(m_RendererID, name.c_str());
	if (location == -1)
		std::cout << "No active uniform variable with name " << name << " found" << std::endl;

	m_UniformLocationCache[name] = location;

	return location;
}



void Shader::SetUniform1f(const std::string& name, float value)
{
	glUniform1f(GetUniformLocation(name), value);
}

void Shader::SetUniform1fv(const std::string& name, std::vector<float> data)
{
	glUniform1fv(GetUniformLocation(name), data.size(), data.data());
}



void Shader::SetUniformUint(const std::string& name, unsigned int value)
{
	glUniform1ui(GetUniformLocation(name), value);
}



void Shader::SetUniform4f(const std::string& name, float f0, float f1, float f2, float f3)
{
	glUniform4f(GetUniformLocation(name), f0, f1, f2, f3);
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
	unsigned int id = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	// Error handling
	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	std::cout << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader compile status: " << result << std::endl;
	if ( result == GL_FALSE )
	{
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char* message = (char*) alloca(length * sizeof(char));
		glGetShaderInfoLog(id, length, &length, message);
		std::cout 
			<< "Failed to compile "
			<< (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
			<< "shader"
			<< std::endl;
		std::cout << message << std::endl;
		glDeleteShader(id);
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

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);

	GLint program_linked;

	glGetProgramiv(program, GL_LINK_STATUS, &program_linked);
	std::cout << "Program link status: " << program_linked << std::endl;
	if (program_linked != GL_TRUE)
	{
		GLsizei log_length = 0;
		GLchar message[1024];
		glGetProgramInfoLog(program, 1024, &log_length, message);
		std::cout << "Failed to link program" << std::endl;
		std::cout << message << std::endl;
	}

	glValidateProgram(program);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}