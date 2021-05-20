// Code taken from "TheCherno" YouTube series on OpenGL
// https://www.youtube.com/watch?v=W3gAzLwfIP0&list=PLlrATfBNZ98foTJPJ_Ev03o2oq3-GGOS2

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

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