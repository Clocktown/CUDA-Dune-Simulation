#include "environment.hpp"
#include <glm/vec3.hpp>
#include <entt/entt.hpp>

namespace sthe
{

// Constructor
Environment::Environment() :
	m_ambientColor{ 1.0f, 1.0f, 1.0f },
	m_ambientIntensity{ 0.25f },
	m_fogMode{ FogMode::None },
	m_fogColor{ 0.4f },
	m_fogDensity{ 0.075f },
	m_fogStart{ 0.0f },
	m_fogEnd{ 22.5f }
{

}

// Setters
void Environment::setAmbientColor(const glm::vec3& t_ambientColor)
{
	m_ambientColor = t_ambientColor;
}

void Environment::setAmbientIntensity(const float t_ambientIntensity)
{
	m_ambientIntensity = t_ambientIntensity;
}

void Environment::setFogMode(const FogMode t_fogMode)
{
	m_fogMode = t_fogMode;
}

void Environment::setFogColor(const glm::vec3& t_fogColor)
{
	m_fogColor = t_fogColor;
}

void Environment::setFogDensity(const float t_fogDensity)
{
	m_fogDensity = t_fogDensity;
}

void Environment::setFogStart(const float t_fogStart)
{
	m_fogStart = t_fogStart;
}

void Environment::setFogEnd(const float t_fogEnd)
{
	m_fogEnd = t_fogEnd;
}

// Getters
const glm::vec3& Environment::getAmbientColor() const
{
	return m_ambientColor;
}

float Environment::getAmbientIntensity() const
{
	return m_ambientIntensity;
}

FogMode Environment::getFogMode() const
{
	return m_fogMode;
}

const glm::vec3& Environment::getFogColor() const
{
	return m_fogColor;
}

float Environment::getFogDensity() const
{
	return m_fogDensity;
}

float Environment::getFogStart() const
{
	return m_fogStart;
}

float Environment::getFogEnd() const
{
	return m_fogEnd;
}

}
