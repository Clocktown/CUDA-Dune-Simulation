#pragma once

#include <sthe/core/component.hpp>
#include <sthe/core/projection.hpp>
#include <sthe/gl/framebuffer.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

namespace sthe
{

class Camera : public Component
{
public:
	// Constructors
	Camera();
	Camera(const Camera& t_camera) = delete;
	Camera(Camera&& t_camera) = default;

	// Destructor
	~Camera() = default;

	// Operators
	Camera& operator=(const Camera& t_camera) = delete;
	Camera& operator=(Camera&& t_camera) = default;
	  
	// Setters
	void setResolution(const glm::ivec2& t_resolution);
	void setAspectRatio(const float t_aspectRatio);
	void setHorizontalFieldOfView(const float t_horizontalFieldOfView);
	void setVerticalFieldOfView(const float t_verticalFieldOfView);
	void setClipPlane(const glm::vec2& t_clipPlane);
	void setProjection(const Projection t_projection);
	void setProjectionMatrix(const glm::mat4& t_projectionMatrix);
	void setFramebuffer(const std::shared_ptr<gl::Framebuffer>& t_framebuffer);
	void setClearColor(const glm::vec4& t_clearColor);
	void setClearMask(const GLbitfield t_clearMask);
	void resetResolution();
	void resetAspectRatio();
	void resetProjectionMatrix();

	// Getters
	const glm::ivec2& getResolution() const;
	float getAspectRatio() const;
	float getHorizontalFieldOfView() const;
	float getVerticalFieldOfView() const;
	const glm::vec2& getClipPlane() const;
	Projection getProjection() const;
	const glm::mat4& getProjectionMatrix() const;
	glm::mat4 getViewMatrix() const;
	glm::mat4 getInverseViewMatrix() const;
	const std::shared_ptr<gl::Framebuffer>& getFramebuffer() const;
	const glm::vec4& getClearColor() const;
	GLbitfield getClearMask() const;
	bool hasFramebuffer() const;
private:	
	// Functionality
	void update() const;
	
	// Attributes
	mutable glm::ivec2 m_resolution;
	mutable float m_aspectRatio;
	float m_verticalFieldOfView;
	glm::vec2 m_clipPlane;
	Projection m_projection;
	mutable glm::mat4 m_projectionMatrix;
	bool m_customResolution;
	bool m_customAspectRatio;
	bool m_customProjectionMatrix;
	mutable bool m_hasChanged;

	std::shared_ptr<gl::Framebuffer> m_framebuffer;
	glm::vec4 m_clearColor;
	GLbitfield m_clearMask;
};

}
