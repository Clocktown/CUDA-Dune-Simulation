#include "shader.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <glad/glad.h>
#include <utility>
#include <string>
#include <vector>

namespace sthe
{
namespace gl
{

// Constructors
Shader::Shader(const GLenum t_type, const std::string& t_file) :
	Shader{ t_type, std::vector<std::string>{ t_file } }
{

}

Shader::Shader(const GLenum t_type, const std::vector<std::string>& t_files) :
	m_handle{ glCreateShader(t_type) },
	m_type{ t_type }
{
	std::vector<std::string> sources;
	std::vector<const char*> pointers;
	std::vector<int> lengths;
	sources.reserve(t_files.size());
	pointers.reserve(t_files.size());
	lengths.reserve(t_files.size());

	for (const std::string& file : t_files)
	{
		const std::string& source{ sources.emplace_back(readFile(file)) };
		pointers.push_back(source.c_str());
		lengths.push_back(static_cast<int>(source.size()));
	}

	GL_CHECK_ERROR(glShaderSource(m_handle, static_cast<GLsizei>(t_files.size()), pointers.data(), lengths.data()));
	GL_CHECK_ERROR(glCompileShader(m_handle));
	GL_CHECK_SHADER(m_handle);
}

Shader::Shader(Shader&& t_shader) noexcept :
	m_handle{ std::exchange(t_shader.m_handle, GL_NONE) },
	m_type{ std::exchange(t_shader.m_type, GL_NONE) }
{

}

// Destructor
Shader::~Shader()
{
	GL_CHECK_ERROR(glDeleteShader(m_handle));
}

// Operator
Shader& Shader::operator=(Shader&& t_shader) noexcept
{
	if (this != &t_shader)
	{
		GL_CHECK_ERROR(glDeleteShader(m_handle));

		m_handle = std::exchange(t_shader.m_handle, GL_NONE);
		m_type = std::exchange(t_shader.m_type, GL_NONE);
	}

	return *this;
}

// Getters
GLuint Shader::getHandle() const
{
	return m_handle;
}

GLenum Shader::getType() const
{
	return m_type;
}

}
}
