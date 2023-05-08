#include "window.hpp"
#include "gui.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>

namespace sthe
{

Window& createWindow(const std::string& t_title, const int t_width, const int t_height)
{
	Window& window{ Window::getInstance(&t_title, t_width, t_height) };
	return window;
}

Window& getWindow()
{
	return Window::getInstance(nullptr, 0, 0);
}

// Static
Window& Window::getInstance(const std::string* const t_title, const int t_width, const int t_height)
{
	static Window window{ t_title, t_width, t_height };
	return window;
}

void Window::sizeCallback(GLFWwindow* const t_window, const int t_width, const int t_height)
{
	Window& window{ getWindow() };
	window.m_size = glm::ivec2{ t_width, t_height };
}

void Window::resolutionCallback(GLFWwindow* const t_window, const int t_resolutionX, const int t_resolutionY)
{
	Window& window{ getWindow() };
	window.m_resolution = glm::ivec2{ t_resolutionX, t_resolutionY };
}

// Constructors
Window::Window(const std::string* const t_title, const int t_width, const int t_height) 	
{
    STHE_ASSERT(t_title != nullptr, "Window was not created");

    [[maybe_unused]] int status{ glfwInit() };
    STHE_ASSERT(status == GLFW_TRUE, "Failed to initialize GLFW");

    m_handle = glfwCreateWindow(t_width, t_height, t_title->c_str(), nullptr, nullptr);
    m_title = *t_title;

    STHE_ASSERT(m_handle != nullptr, "Failed to create window");

    GLFW_CHECK_ERROR(glfwSetWindowUserPointer(m_handle, this));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(m_handle, sizeCallback));
    GLFW_CHECK_ERROR(glfwSetFramebufferSizeCallback(m_handle, resolutionCallback));

    GLFW_CHECK_ERROR(glfwGetWindowSize(m_handle, &m_size.x, &m_size.y));
    GLFW_CHECK_ERROR(glfwGetFramebufferSize(m_handle, &m_resolution.x, &m_resolution.y));

    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_DOUBLEBUFFER, 1));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_DEPTH_BITS, 24));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_STENCIL_BITS, 8));

    GLFW_CHECK_ERROR(glfwMakeContextCurrent(m_handle));

    status = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    STHE_ASSERT(status == GLFW_TRUE, "Failed to load OpenGL functionality");
}

// Destructor
Window::~Window()
{
	GLFW_CHECK_ERROR(glfwTerminate());
}

// Functionality
void Window::update()
{
	GLFW_CHECK_ERROR(glfwSwapBuffers(m_handle));
	GLFW_CHECK_ERROR(glfwPollEvents());
}

// Setter
void Window::setTitle(const std::string& t_title)
{
	m_title = t_title;
	GLFW_CHECK_ERROR(glfwSetWindowTitle(m_handle, m_title.c_str()));
}

// Getters
GLFWwindow* Window::getHandle() const
{
	return m_handle;
}

const std::string& Window::getTitle() const
{
	return m_title;
}

const glm::ivec2& Window::getSize() const
{
	return m_size;
}

const glm::ivec2& Window::getResolution() const
{
	return m_resolution;
}

bool Window::isOpen() const
{
	return glfwWindowShouldClose(m_handle) == GLFW_FALSE;
}

}
