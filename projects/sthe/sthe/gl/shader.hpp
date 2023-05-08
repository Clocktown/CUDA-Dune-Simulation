#pragma once

#include "handle.hpp"
#include <glad/glad.h>
#include <string>
#include <vector>

namespace sthe
{
namespace gl
{

class Shader : public Handle
{
public:
	// Constructors
	Shader(const GLenum t_type, const std::string& t_file);
	Shader(const GLenum t_type, const std::vector<std::string>& t_files);
	Shader(const Shader& t_shader) = delete;
	Shader(Shader&& t_shader) noexcept;
	 
	// Destructor
	~Shader();

	// Operators
	Shader& operator=(const Shader& t_shader) = delete;
	Shader& operator=(Shader&& t_shader) noexcept;

	// Getters
	GLuint getHandle() const override;
	GLenum getType() const;
private:
	// Attributes
	GLuint m_handle;
	GLenum m_type;
};

}
}
