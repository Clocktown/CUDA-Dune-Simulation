#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>

namespace sthe
{

class Window
{
public:
	// Constructors
	Window(const Window& t_window) = delete;
	Window(Window&& t_window) = delete;

	// Destructor
	~Window();

	// Operators
	Window& operator=(const Window& t_window) = delete;
	Window& operator=(Window&& t_window) = delete;

	// Functionality
	void update();
	
	// Setter
	void setTitle(const std::string& t_title);

	// Getters
	GLFWwindow* getHandle() const;
	const std::string& getTitle() const;
	const glm::ivec2& getSize() const;
	const glm::ivec2& getResolution() const;
	bool isOpen() const;
private:
	// Static
	static Window& getInstance(const std::string* const t_title, const int t_width, const int t_height);
	static void sizeCallback(GLFWwindow* const t_window, const int t_width, const int t_height);
	static void resolutionCallback(GLFWwindow* const t_window, const int t_framebufferWidth, const int t_framebufferHeight);

	// Constructor
	Window(const std::string* const t_title, const int t_width, const int t_height);

	// Attributes
	GLFWwindow* m_handle;
	std::string m_title;
	glm::ivec2 m_size;
	glm::ivec2 m_resolution;

	// Friends
	friend Window& createWindow(const std::string& t_title, const int t_width, const int t_height);
	friend Window& getWindow();
};

Window& createWindow(const std::string& t_title, const int t_width, const int t_height);
Window& getWindow();

}
