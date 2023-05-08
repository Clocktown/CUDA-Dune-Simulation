#include "debug.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <vector>

namespace sthe
{
namespace detail
{

void glfwCheckError(const char* const t_file, const int t_line)
{
	const char* description;

	if (glfwGetError(&description) != GLFW_NO_ERROR)
	{
		std::cerr << "GLFW Error: " + std::string{ description } + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckError(const char* const t_file, const int t_line)
{
	const GLenum error{ glGetError() };

	if (error != GL_NO_ERROR)
	{
		std::cerr << "OpenGL Error: " + std::to_string(error) + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckProgram(const GLuint t_program, const char* const t_file, const int t_line)
{
	GLint status;
	GL_CHECK_ERROR(glGetProgramiv(t_program, GL_LINK_STATUS, &status));

	if (status == GL_FALSE)
	{
		GLint length;
		GL_CHECK_ERROR(glGetProgramiv(t_program, GL_INFO_LOG_LENGTH, &length));

		std::vector<GLchar> log(length);
		GL_CHECK_ERROR(glGetProgramInfoLog(t_program, length, nullptr, log.data()));

		std::cerr << "OpenGL Error: " + std::string{ log.data(), static_cast<size_t>(length) } + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckShader(const GLuint t_shader, const char* const t_file, const int t_line)
{
	GLint status;
	GL_CHECK_ERROR(glGetShaderiv(t_shader, GL_COMPILE_STATUS, &status));

	if (status == GL_FALSE)
	{
		GLint length;
		GL_CHECK_ERROR(glGetShaderiv(t_shader, GL_INFO_LOG_LENGTH, &length));

		std::vector<GLchar> log(length);
		glGetShaderInfoLog(t_shader, length, nullptr, log.data());

		std::cerr << "OpenGL Error: " + std::string{ log.data(), static_cast<size_t>(length) } + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void glCheckFramebuffer(const GLuint t_framebuffer, const GLenum t_target, const char* const t_file, const int t_line)
{
	if (glCheckNamedFramebufferStatus(t_framebuffer, t_target) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "OpenGL Error: Framebuffer is not complete\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void cuCheckError(const cudaError_t t_error, const char* const t_file, const int t_line)
{
	if (t_error != cudaSuccess)
	{
		std::cerr << "CUDA Error: " + std::string{ cudaGetErrorString(t_error) } + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

void curandCheckError(const curandStatus_t t_error, const char* const t_file, const int t_line)
{
	if (t_error != cudaSuccess)
	{
		std::cerr << "CURAND Error : " + std::to_string(static_cast<int>(t_error))  + "\n"
			      << "File: " + std::string{ t_file } + "\n"
			      << "Line: " + std::to_string(t_line) + "\n";

		std::exit(EXIT_FAILURE);
	}
}

}
}
