#pragma once

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <curand.h>

#ifdef NDEBUG
#   define STHE_ERROR(t_message) static_cast<void>(0)
#   define STHE_ASSERT(t_condition, t_message) static_cast<void>(0)
#   define GLFW_CHECK_ERROR(t_function) t_function
#   define GL_CHECK_ERROR(t_function) t_function
#   define GL_CHECK_SHADER(t_shader) static_cast<void>(0)
#   define GL_CHECK_PROGRAM(t_program) static_cast<void>(0)
#   define GL_CHECK_FRAMEBUFFER(t_framebuffer, t_target) static_cast<void>(0)
#   define CU_CHECK_ERROR(t_function) t_function
#   define CURAND_CHECK_ERROR(t_function) t_function
#else
#   include <iostream>
#   include <string>
#   define STHE_ERROR(t_message) std::cerr << "STHE Error: " + std::string{ t_message } + "\n"\
                                           << "File: " + std::string{ __FILE__ } + "\n"\
                                           << "Line: " + std::to_string(__LINE__) + "\n";\
                                 \
                                 std::exit(EXIT_FAILURE)
#   define STHE_ASSERT(t_condition, t_message) if (!(t_condition))\
                                                {\
                                                    STHE_ERROR(t_message);\
                                                }\
                                                static_cast<void>(0)
#   define GLFW_CHECK_ERROR(t_function) t_function; sthe::detail::glfwCheckError(__FILE__, __LINE__) 
#   define GL_CHECK_ERROR(t_function) t_function; sthe::detail::glCheckError(__FILE__, __LINE__) 
#   define GL_CHECK_SHADER(t_shader) sthe::detail::glCheckShader(t_shader, __FILE__, __LINE__) 
#   define GL_CHECK_PROGRAM(t_program) sthe::detail::glCheckProgram(t_program, __FILE__, __LINE__) 
#   define GL_CHECK_FRAMEBUFFER(t_framebuffer, t_target) sthe::detail::glCheckFramebuffer(t_framebuffer, t_target, __FILE__, __LINE__) 
#   define CU_CHECK_ERROR(t_function) sthe::detail::cuCheckError(t_function, __FILE__, __LINE__)
#   define CURAND_CHECK_ERROR(t_function) sthe::detail::curandCheckError(t_function, __FILE__, __LINE__)
#endif

namespace sthe
{
namespace detail
{

void glfwCheckError(const char* const t_file, const int t_line);
void glCheckError(const char* const t_file, const int t_line);
void glCheckProgram(const GLuint t_program, const char* const t_file, const int t_line);
void glCheckShader(const GLuint t_shader, const char* const t_file, const int t_line);
void glCheckFramebuffer(const GLuint t_framebuffer, const GLenum t_target, const char* const t_file, const int t_line);
void cuCheckError(const cudaError_t t_error, const char* const t_file, const int t_line);
void curandCheckError(const curandStatus_t t_error, const char* const t_file, const int t_line);

}
}
