#pragma once

#include <sthe/components/light.hpp>
#include <glm/glm.hpp>

namespace sthe
{

enum class FogMode : unsigned char
{
	None,
	Linear,
	Exponential,
	Exponential2
};

namespace uniform
{

struct Environment
{
	glm::vec3 ambientColor;
	float ambientIntensity;
	glm::vec3 fogColor;
	float fogDensity;
	unsigned int fogMode;
	float fogStart;
	float fogEnd;
	int lightCount;
	std::array<Light, 16> lights;
};

}

class Environment
{
public:
	// Constructors
	Environment();
	Environment(const Environment& t_environment) = default;
	Environment(Environment&& t_environment) = default;

	// Destructor
	~Environment() = default;

	// Operators
	Environment& operator=(const Environment& t_environment) = default;
	Environment& operator=(Environment&& t_environment) = default;

	// Setters
	void setAmbientColor(const glm::vec3& t_ambientColor);
	void setAmbientIntensity(const float t_ambientIntensity);
	void setFogMode(const FogMode t_fogMode);
	void setFogColor(const glm::vec3& t_fogColor);
	void setFogDensity(const float t_fogDensity);
	void setFogStart(const float t_fogStart);
	void setFogEnd(const float t_fogEnd);

	// Getters
	const glm::vec3& getAmbientColor() const;
	float getAmbientIntensity() const;
	FogMode getFogMode() const;
	const glm::vec3& getFogColor() const;
	float getFogDensity() const;
	float getFogStart() const;
	float getFogEnd() const;
private:
	// Attributes
	glm::vec3 m_ambientColor;
	float m_ambientIntensity;
	FogMode m_fogMode;
	glm::vec3 m_fogColor;
	float m_fogDensity;
	float m_fogStart;
	float m_fogEnd;
};

}
