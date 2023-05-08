#pragma once

#include "handle.hpp"
#include "shader.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace sthe
{
namespace gl
{

class Program : public Handle
{
public:
	// Constructors
	Program();
	Program(const Program& t_program) = delete;
	Program(Program&& t_program) noexcept;

	// Destructor
	~Program();

	// Operators
	Program& operator=(const Program& t_program) = delete;
	Program& operator=(Program&& t_program) noexcept;
	
	// Functionality
	void use() const;
	void disuse() const;
	void dispatch(const int t_gridSize) const;
	void dispatch(const glm::ivec2& t_gridSize) const;
	void dispatch(const glm::ivec3& t_gridSize) const;
	void link(const bool t_detachAll = true);
	void attachShader(const Shader& t_shader);
	void attachShader(const GLuint t_shader);
	void detachShader(const Shader& t_shader);
	void detachShader(const GLuint t_shader);
	void detachAll();
	
	// Setters
	void setPatchVertexCount(const int t_patchVertexCount);
	void setUniform(const int t_location, const bool t_value);
	void setUniform(const int t_location, const int t_value);
	void setUniform(const int t_location, const unsigned int t_value);
	void setUniform(const int t_location, const float t_value);
	void setUniform(const int t_location, const double t_value);
	void setUniform(const int t_location, const glm::bvec2& t_value);
	void setUniform(const int t_location, const glm::ivec2& t_value);
	void setUniform(const int t_location, const glm::uvec2& t_value);
	void setUniform(const int t_location, const glm::vec2& t_value);
	void setUniform(const int t_location, const glm::dvec2& t_value);
	void setUniform(const int t_location, const glm::bvec3& t_value);
	void setUniform(const int t_location, const glm::ivec3& t_value);
	void setUniform(const int t_location, const glm::uvec3& t_value);
	void setUniform(const int t_location, const glm::vec3& t_value);
	void setUniform(const int t_location, const glm::dvec3& t_value);
	void setUniform(const int t_location, const glm::bvec4& t_value);
	void setUniform(const int t_location, const glm::ivec4& t_value);
	void setUniform(const int t_location, const glm::uvec4& t_value);
	void setUniform(const int t_location, const glm::vec4& t_value);
	void setUniform(const int t_location, const glm::dvec4& t_value);
	void setUniform(const int t_location, const glm::mat2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat2x3& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat2x3& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat2x4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat2x4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat3& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat3& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat3x2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat3x2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat3x4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat3x4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat4& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat4x2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat4x2& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::mat4x3& t_value, const bool t_transpose = false);
	void setUniform(const int t_location, const glm::dmat4x3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const bool t_value);
	void setUniform(const std::string& t_name, const int t_value);
	void setUniform(const std::string& t_name, const unsigned int t_value);
	void setUniform(const std::string& t_name, const float t_value);
	void setUniform(const std::string& t_name, const double t_value);
	void setUniform(const std::string& t_name, const glm::bvec2& t_value);
	void setUniform(const std::string& t_name, const glm::ivec2& t_value);
	void setUniform(const std::string& t_name, const glm::uvec2& t_value);
	void setUniform(const std::string& t_name, const glm::vec2& t_value);
	void setUniform(const std::string& t_name, const glm::dvec2& t_value);
	void setUniform(const std::string& t_name, const glm::bvec3& t_value);
	void setUniform(const std::string& t_name, const glm::ivec3& t_value);
	void setUniform(const std::string& t_name, const glm::uvec3& t_value);
	void setUniform(const std::string& t_name, const glm::vec3& t_value);
	void setUniform(const std::string& t_name, const glm::dvec3& t_value);
	void setUniform(const std::string& t_name, const glm::bvec4& t_value);
	void setUniform(const std::string& t_name, const glm::ivec4& t_value);
	void setUniform(const std::string& t_name, const glm::uvec4& t_value);
	void setUniform(const std::string& t_name, const glm::vec4& t_value);
	void setUniform(const std::string& t_name, const glm::dvec4& t_value);
	void setUniform(const std::string& t_name, const glm::mat2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat2x3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat2x3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat2x4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat2x4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat3x2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat3x2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat3x4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat3x4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat4& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat4x2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat4x2& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::mat4x3& t_value, const bool t_transpose = false);
	void setUniform(const std::string& t_name, const glm::dmat4x3& t_value, const bool t_transpose = false);

	// Getters
	GLuint getHandle() const override;
	int getPatchVertexCount() const;
	int getUniformLocation(const std::string& t_name) const;
private:
	// Attribute
	GLuint m_handle;
	int m_patchVertexCount;
};

}
}
