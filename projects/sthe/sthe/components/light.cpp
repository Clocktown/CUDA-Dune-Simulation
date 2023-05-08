#include "light.hpp"
#include <sthe/config/debug.hpp>
#include <glm/glm.hpp>

namespace sthe
{


glm::vec3 getSeamlessAttenuation(const float t_range)
{
	return glm::vec3{ 1.0f, 2.0f / t_range, 9.0f / (t_range * t_range) };
}

// Constructor
Light::Light(const LightType t_type) :
	m_type{ t_type },
	m_color{ 1.0f },
	m_intensity{ 1.0f },
	m_range{ 40.0f },
	m_attenuation{ getSeamlessAttenuation(40.0f) },
	m_spotAngle{ 45.0f },
	m_spotBlend{ 0.15f },
	m_customAttenuation{ false }
{
}

// Setters
void Light::setType(const LightType t_type)
{
	m_type = t_type;
}

void Light::setColor(const glm::vec3& t_color)
{
	m_color = t_color;
}

void Light::setIntensity(const float t_intensity)
{
	m_intensity = t_intensity;
}

void Light::setAttenuation(const glm::vec3& t_attenuation)
{
	m_attenuation = t_attenuation;
	m_customAttenuation = true;
}

void Light::setRange(const float t_range)
{
	STHE_ASSERT(t_range != 0.0f, "Range cannot be equal to 0");

	m_range = t_range;

	if (!m_customAttenuation)
	{
		m_attenuation = getSeamlessAttenuation(m_range);
	}
}

void Light::setSpotAngle(const float t_spotAngle)
{
	m_spotAngle = t_spotAngle;
}

void Light::setSpotBlend(const float t_spotBlend)
{
	m_spotBlend = t_spotBlend;
}

void Light::resetAttenuation()
{
	m_attenuation = getSeamlessAttenuation(m_range);
	m_customAttenuation = false;
}

// Getters
LightType Light::getType() const
{
	return m_type;
}

const glm::vec3& Light::getColor() const
{
	return m_color;
}

float Light::getIntensity() const
{
	return m_intensity;
}

const glm::vec3& Light::getAttenuation() const
{
	return m_attenuation;
}

float Light::getRange() const
{
	return m_range;
}

float Light::getSpotAngle() const
{
	return m_spotAngle;
}

float Light::getSpotBlend() const
{
	return m_spotBlend;
}

}
