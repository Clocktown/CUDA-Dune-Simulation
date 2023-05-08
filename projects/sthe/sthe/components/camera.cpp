#include "camera.hpp"
#include "transform.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/core/window.hpp>
#include <sthe/core/game_object.hpp>
#include <sthe/core/component.hpp>
#include <sthe/gl/framebuffer.hpp>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

namespace sthe
{

// Constructor
Camera::Camera() :
	m_resolution{ 0, 0 },
	m_aspectRatio{ 0.0f },
	m_verticalFieldOfView{ 60.0f },
	m_clipPlane{ 0.1f, 1000.0f },
	m_projection{ Projection::Perspective },
	m_projectionMatrix{ 1.0f },
	m_customResolution{ false },
	m_customAspectRatio{ false },
	m_customProjectionMatrix{ false },
	m_hasChanged{ true },
	m_framebuffer{ nullptr },
	m_clearColor{ 0.7f, 0.9f, 1.0f, 1.0f },
	m_clearMask{ GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT }
{
	
}

// Functionality
void Camera::update() const
{
	if (!m_customResolution)
	{
		glm::ivec2 resolution;

		if (hasFramebuffer())
		{
			resolution.x = m_framebuffer->getWidth();
			resolution.y = m_framebuffer->getHeight();
		}
		else
		{
			const Window& window{ getWindow() };
			resolution = window.getResolution();
		}

		if (m_resolution != resolution)
		{
			m_resolution = resolution;
		
			if (!m_customAspectRatio)
			{
				STHE_ASSERT(m_resolution.y != 0, "Resolution y cannot be than 0");

				m_aspectRatio = static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y);
			}

			m_hasChanged = true;
		}
	}
}

// Setters
void Camera::setResolution(const glm::ivec2& t_resolution)
{
	m_resolution = t_resolution;

	if (!m_customAspectRatio)
	{
		STHE_ASSERT(m_resolution.y != 0, "Resolution y cannot be than 0");

		m_aspectRatio = static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y);
	}

	m_customResolution = true;
	m_hasChanged = true;
}

void Camera::setAspectRatio(const float t_aspectRatio)
{
	m_aspectRatio = t_aspectRatio;
	m_customAspectRatio = true;
	m_hasChanged = true;
}

void Camera::setHorizontalFieldOfView(const float t_horizontalFieldOfView)
{
	setVerticalFieldOfView(glm::degrees(2.0f * glm::atan(glm::tan(0.5f * glm::radians(t_horizontalFieldOfView)) / getAspectRatio())));
}

void Camera::setVerticalFieldOfView(const float t_verticalFieldOfView)
{
	m_verticalFieldOfView = t_verticalFieldOfView;
	m_hasChanged = true;
}

void Camera::setClipPlane(const glm::vec2& t_clipPlane)
{
	m_clipPlane = t_clipPlane;
	m_hasChanged = true;
}

void Camera::setProjection(const Projection t_projection)
{
	m_projection = t_projection;
	m_hasChanged = true;
}

void Camera::setProjectionMatrix(const glm::mat4& t_projectionMatrix)
{
	m_projectionMatrix = t_projectionMatrix;
	m_customProjectionMatrix = true;
}

void Camera::setFramebuffer(const std::shared_ptr<gl::Framebuffer>& t_framebuffer)
{
	m_framebuffer = t_framebuffer;
}

void Camera::setClearColor(const glm::vec4& t_clearColor)
{
	m_clearColor = t_clearColor;
}

void Camera::setClearMask(const GLbitfield t_clearMask)
{
	m_clearMask = t_clearMask;
}

void Camera::resetResolution()
{
	m_customResolution = false;
	m_hasChanged = true;
}

void Camera::resetAspectRatio()
{
	STHE_ASSERT(m_resolution.y != 0.0f, "Resolution y cannot be equal to 0");

	m_aspectRatio = static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y);
	m_customAspectRatio = false;
	m_hasChanged = true;
}

void Camera::resetProjectionMatrix()
{
	m_customProjectionMatrix = false;
	m_hasChanged = true;
}

// Getters
const glm::ivec2& Camera::getResolution() const
{
	update();
	return m_resolution;
}

float Camera::getAspectRatio() const
{
	update();
	return m_aspectRatio;
}

float Camera::getVerticalFieldOfView() const
{
	return m_verticalFieldOfView;
}

float Camera::getHorizontalFieldOfView() const
{
	return glm::degrees(2.0f * glm::atan(glm::tan(0.5f * glm::radians(m_verticalFieldOfView)) * getAspectRatio()));
}

const glm::vec2& Camera::getClipPlane() const
{
	return m_clipPlane;
}

Projection Camera::getProjection() const
{
	return m_projection;
}

const glm::mat4& Camera::getProjectionMatrix() const
{
	if (!m_customProjectionMatrix)
	{
		update();

		if (m_hasChanged)
		{
			if (m_projection == Projection::Perspective)
			{
				m_projectionMatrix = glm::perspective(glm::radians(m_verticalFieldOfView), m_aspectRatio, m_clipPlane.x, m_clipPlane.y);
			}
			else
			{
				m_projectionMatrix = glm::ortho(0.0f, static_cast<float>(m_resolution.x), 0.0f, static_cast<float>(m_resolution.y));
			}

			m_hasChanged = false;
		}
	}

	return m_projectionMatrix;
}

glm::mat4 Camera::getViewMatrix() const
{
	const Transform& transform{ getGameObject().getTransform() };
	return glm::translate(glm::mat4_cast(glm::conjugate(transform.getRotation())), -transform.getPosition());
}

glm::mat4 Camera::getInverseViewMatrix() const
{
	const Transform& transform{ getGameObject().getTransform() };

	glm::mat4 inverseViewMatrix{ glm::mat4_cast(transform.getRotation()) };
	inverseViewMatrix[3] = glm::vec4{ transform.getPosition(), 1.0f };

	return inverseViewMatrix;
}

const glm::vec4& Camera::getClearColor() const
{
	return m_clearColor;
}

GLbitfield Camera::getClearMask() const
{
	return m_clearMask;
}

const std::shared_ptr<gl::Framebuffer>& Camera::getFramebuffer() const
{
	return m_framebuffer;
}

bool Camera::hasFramebuffer() const
{
	return m_framebuffer != nullptr;
}

}
