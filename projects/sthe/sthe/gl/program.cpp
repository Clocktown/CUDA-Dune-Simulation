#include "program.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <string>
#include <vector>

namespace sthe
{
namespace gl
{

// Constructors
Program::Program() :
	m_handle{ glCreateProgram() },
	m_patchVertexCount{ 0 }
{

}

Program::Program(Program&& t_program) noexcept :
	m_handle{ std::exchange(t_program.m_handle, GL_NONE) },
	m_patchVertexCount{ std::exchange(t_program.m_patchVertexCount, 0) }
{

}

// Destructor
Program::~Program()
{
	GL_CHECK_ERROR(glDeleteProgram(m_handle));
}

// Operator
Program& Program::operator=(Program&& t_program) noexcept
{
	if (this != &t_program)
	{
		GL_CHECK_ERROR(glDeleteProgram(m_handle));
		m_handle = std::exchange(t_program.m_handle, GL_NONE);
		m_patchVertexCount = std::exchange(t_program.m_patchVertexCount, 0);
	}

	return *this;
}

// Functionality
void Program::use() const
{
	GL_CHECK_ERROR(glUseProgram(m_handle));

	if (m_patchVertexCount > 0)
	{
		GL_CHECK_ERROR(glPatchParameteri(GL_PATCH_VERTICES, m_patchVertexCount));
	}
}

void Program::disuse() const
{
	GL_CHECK_ERROR(glUseProgram(GL_NONE));
}

void Program::dispatch(const int t_gridSize) const
{
	dispatch(glm::ivec3{ t_gridSize, 1, 1 });
}

void Program::dispatch(const glm::ivec2& t_gridSize) const
{
	dispatch(glm::ivec3{ t_gridSize, 1 });
}

void Program::dispatch(const glm::ivec3& t_gridSize) const
{
	STHE_ASSERT(t_gridSize.x >= 0, "Grid size x must be greater than or equal to 0");
	STHE_ASSERT(t_gridSize.y >= 0, "Grid size y must be greater than or equal to 0");
	STHE_ASSERT(t_gridSize.z >= 0, "Grid size z must be greater than or equal to 0");

	GL_CHECK_ERROR(glUseProgram(m_handle));
	GL_CHECK_ERROR(glDispatchCompute(static_cast<GLuint>(t_gridSize.x), static_cast<GLuint>(t_gridSize.y), static_cast<GLuint>(t_gridSize.z)));
}

void Program::link(const bool t_detachAll)
{
	GL_CHECK_ERROR(glLinkProgram(m_handle));
	GL_CHECK_PROGRAM(m_handle);

	if (t_detachAll)
	{
		detachAll();
	}
}

void Program::attachShader(const Shader& t_shader)
{
	GL_CHECK_ERROR(glAttachShader(m_handle, t_shader.getHandle()));
}

void Program::attachShader(const GLuint t_shader)
{
	GL_CHECK_ERROR(glAttachShader(m_handle, t_shader));
}

void Program::detachShader(const Shader& t_shader)
{
	GL_CHECK_ERROR(glDetachShader(m_handle, t_shader.getHandle()));
}

void Program::detachShader(const GLuint t_shader)
{
	GL_CHECK_ERROR(glDetachShader(m_handle, t_shader));
}

void Program::detachAll()
{
	GLint shaderCount;
	GL_CHECK_ERROR(glGetProgramiv(m_handle, GL_ATTACHED_SHADERS, &shaderCount));

	std::vector<GLuint> shaders(shaderCount);
	GL_CHECK_ERROR(glGetAttachedShaders(m_handle, shaderCount, nullptr, shaders.data()));

	for (const GLuint shaderHandle : shaders)
	{
		GL_CHECK_ERROR(glDetachShader(m_handle, shaderHandle));
	}
}

// Setters
void Program::setPatchVertexCount(const int t_patchVertexCount)
{
	m_patchVertexCount = t_patchVertexCount;
}

void Program::setUniform(const int t_location, const bool t_value)
{
	GL_CHECK_ERROR(glProgramUniform1i(m_handle, t_location, t_value));
}

void Program::setUniform(const int t_location, const int t_value)
{
	GL_CHECK_ERROR(glProgramUniform1i(m_handle, t_location, t_value));
}

void Program::setUniform(const int t_location, const unsigned int t_value)
{
	GL_CHECK_ERROR(glProgramUniform1ui(m_handle, t_location, t_value));
}

void Program::setUniform(const int t_location, const float t_value)
{
	GL_CHECK_ERROR(glProgramUniform1f(m_handle, t_location, t_value));
}

void Program::setUniform(const int t_location, const double t_value)
{
	GL_CHECK_ERROR(glProgramUniform1d(m_handle, t_location, t_value));
}

void Program::setUniform(const int t_location, const glm::bvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2i(m_handle, t_location, t_value.x, t_value.y));
}

void Program::setUniform(const int t_location, const glm::ivec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2i(m_handle, t_location, t_value.x, t_value.y));
}

void Program::setUniform(const int t_location, const glm::uvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2ui(m_handle, t_location, t_value.x, t_value.y));
}

void Program::setUniform(const int t_location, const glm::vec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2f(m_handle, t_location, t_value.x, t_value.y));
}

void Program::setUniform(const int t_location, const glm::dvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2d(m_handle, t_location, t_value.x, t_value.y));
}

void Program::setUniform(const int t_location, const glm::bvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3i(m_handle, t_location, t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const int t_location, const glm::ivec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3i(m_handle, t_location, t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const int t_location, const glm::uvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3ui(m_handle, t_location, t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const int t_location, const glm::vec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3f(m_handle, t_location, t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const int t_location, const glm::dvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3d(m_handle, t_location, t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const int t_location, const glm::bvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4i(m_handle, t_location, t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const int t_location, const glm::ivec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4i(m_handle, t_location, t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const int t_location, const glm::uvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4ui(m_handle, t_location, t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const int t_location, const glm::vec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4f(m_handle, t_location, t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const int t_location, const glm::dvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4d(m_handle, t_location, t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const int t_location, const glm::mat2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat2x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x3fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat2x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x3dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat2x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x4fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat2x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x4dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat3x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x2fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat3x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x2dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat3x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x4fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat3x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x4dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat4x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x2fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat4x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x2dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::mat4x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x3fv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const int t_location, const glm::dmat4x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x3dv(m_handle, t_location, 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const bool t_value)
{
	GL_CHECK_ERROR(glProgramUniform1i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value));
}

void Program::setUniform(const std::string& t_name, const int t_value)
{
	GL_CHECK_ERROR(glProgramUniform1i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value));
}

void Program::setUniform(const std::string& t_name, const unsigned int t_value)
{
	GL_CHECK_ERROR(glProgramUniform1ui(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value));
}

void Program::setUniform(const std::string& t_name, const float t_value)
{
	GL_CHECK_ERROR(glProgramUniform1f(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value));
}

void Program::setUniform(const std::string& t_name, const double t_value)
{
	GL_CHECK_ERROR(glProgramUniform1d(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value));
}

void Program::setUniform(const std::string& t_name, const glm::bvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y));
}

void Program::setUniform(const std::string& t_name, const glm::ivec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y));
}

void Program::setUniform(const std::string& t_name, const glm::uvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2ui(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y));
}

void Program::setUniform(const std::string& t_name, const glm::vec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2f(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y));
}

void Program::setUniform(const std::string& t_name, const glm::dvec2& t_value)
{
	GL_CHECK_ERROR(glProgramUniform2d(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y));
}

void Program::setUniform(const std::string& t_name, const glm::bvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const std::string& t_name, const glm::ivec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const std::string& t_name, const glm::uvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3ui(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const std::string& t_name, const glm::vec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3f(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const std::string& t_name, const glm::dvec3& t_value)
{
	GL_CHECK_ERROR(glProgramUniform3d(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z));
}

void Program::setUniform(const std::string& t_name, const glm::bvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const std::string& t_name, const glm::ivec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4i(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const std::string& t_name, const glm::uvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4ui(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const std::string& t_name, const glm::vec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4f(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const std::string& t_name, const glm::dvec4& t_value)
{
	GL_CHECK_ERROR(glProgramUniform4d(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), t_value.x, t_value.y, t_value.z, t_value.w));
}

void Program::setUniform(const std::string& t_name, const glm::mat2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat2x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x3fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat2x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x3dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat2x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x4fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat2x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix2x4dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat3x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x2fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat3x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x2dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat3x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x4fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat3x4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix3x4dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat4& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat4x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x2fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat4x2& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x2dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::mat4x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x3fv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

void Program::setUniform(const std::string& t_name, const glm::dmat4x3& t_value, const bool t_transpose)
{
	GL_CHECK_ERROR(glProgramUniformMatrix4x3dv(m_handle, glGetUniformLocation(m_handle, t_name.c_str()), 1, t_transpose, glm::value_ptr(t_value)));
}

// Getters
GLuint Program::getHandle() const
{
	return m_handle;
}

int Program::getPatchVertexCount() const
{
	return m_patchVertexCount;
}

int Program::getUniformLocation(const std::string& t_name) const
{
	return glGetUniformLocation(m_handle, t_name.c_str());
}

}
}
