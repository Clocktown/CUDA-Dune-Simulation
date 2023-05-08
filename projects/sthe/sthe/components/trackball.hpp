#pragma once

#include <sthe/core/component.hpp>
#include <glm/glm.hpp>

namespace sthe
{

class Trackball : public Component
{
public:
	// Constructors
	explicit Trackball(const glm::vec3 t_target = glm::vec3{ 0.0f });
	Trackball(const Trackball& t_trackball) = delete;
	Trackball(Trackball&& t_trackball) = default;

	// Destructor
	~Trackball() = default;

	// Operators
	Trackball& operator=(const Trackball& t_trackball) = delete;
	Trackball& operator=(Trackball&& t_trackball) = default;

	// Functionality
	void lateUpdate();
	
	// Setters
	void setTarget(const glm::vec3& t_target);
	void setPanSensitivity(const float t_panSensitivity);
	void setOrbitSensitivity(const float t_orbitSensitivity);
	void setZoomSensitivity(const float t_zoomSensitivity);

	// Getters
	const glm::vec3& getTarget() const;
	float getPanSensitivity() const;
	float getOrbitSensitivity() const;
	float getZoomSensitivity() const;
private:
	glm::vec3 m_target;
	float m_panSensitivity;
	float m_orbitsensitivity;
	float m_zoomSensitivity;
};

}
