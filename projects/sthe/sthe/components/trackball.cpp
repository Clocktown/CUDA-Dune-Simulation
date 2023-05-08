#include "trackball.hpp"
#include "transform.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/core/application.hpp>
#include <sthe/core/game_object.hpp>
#include <sthe/core/world.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>

namespace sthe
{

// Constructor
Trackball::Trackball(const glm::vec3 t_target) :
	m_target{ t_target },
	m_panSensitivity{ 0.01f },
	m_orbitsensitivity{ 0.15f },
	m_zoomSensitivity{ 0.75f }
{

}

// Functionality
void Trackball::lateUpdate()
{
	const ImGuiIO& io{ ImGui::GetIO() };
	const glm::vec2 mouseDelta{ io.MouseDelta.x, io.MouseDelta.y };

	Transform& transform{ getGameObject().getTransform() };
	transform.lookAt(m_target);

	if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
	{
		const glm::vec3 translation{ -m_panSensitivity * (mouseDelta.x * transform.getRight() - mouseDelta.y * transform.getUp()) };

		transform.translate(translation, Space::World);
		m_target += translation;
	}

	if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
	{
		const glm::vec3& distanceVector{ m_target - transform.getPosition() };
		const float distance{ glm::length(distanceVector) };

		const float pitch{ glm::degrees(glm::asin(distanceVector.y / (distance + glm::epsilon<float>()))) };
		float pitchDelta{ -m_orbitsensitivity * mouseDelta.y };
		pitchDelta = glm::clamp(pitch + pitchDelta, -89.0f, 89.0f) - pitch;

		const float yawDelta{ -m_orbitsensitivity * mouseDelta.x };

		transform.translate(distanceVector, Space::World);
		transform.rotate(pitchDelta, World::right);
		transform.rotate(yawDelta, World::up, Space::World);
		transform.translate(-distance * World::forward);
	}

	const float scrollDelta{ io.MouseWheel };

	if (scrollDelta != 0.0f)
	{
		const float distance{ glm::length(m_target - transform.getPosition()) };
		const glm::vec3 translation{ glm::min(m_zoomSensitivity * scrollDelta, distance - 0.5f) * transform.getForward() };

		transform.translate(translation, Space::World);
	}
}

// Setters
void Trackball::setTarget(const glm::vec3& t_target)
{
	m_target = t_target;
}

void Trackball::setPanSensitivity(const float t_panSensitivity)
{
	m_panSensitivity = t_panSensitivity;
}

void Trackball::setOrbitSensitivity(const float t_sensitivity)
{
	m_orbitsensitivity = t_sensitivity;
}

void Trackball::setZoomSensitivity(const float t_zoomSensitivity)
{
	m_zoomSensitivity = t_zoomSensitivity;
}

// Getters
const glm::vec3& Trackball::getTarget() const
{
	return m_target;
}

float Trackball::getPanSensitivity() const
{
	return m_panSensitivity;
}

float Trackball::getOrbitSensitivity() const
{
	return m_orbitsensitivity;
}

float Trackball::getZoomSensitivity() const
{
	return m_zoomSensitivity;
}

}
