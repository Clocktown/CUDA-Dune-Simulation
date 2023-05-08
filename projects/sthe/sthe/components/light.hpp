#pragma once

#include <sthe/core/component.hpp>
#include <glm/glm.hpp>

namespace sthe
{

glm::vec3 getSeamlessAttenuation(const float t_range);

enum class LightType : unsigned char
{
	Point,
	Spot,
	Directional
};

namespace uniform
{

struct Light
{
	glm::vec3 position;
	unsigned int type;
	glm::vec3 color;
	float intensity;
	glm::vec3 attenuation;
	float range;
	glm::vec3 direction;
	float spotOuterCutOff; // Cosine of the outer angle
	float spotInnerCutOff; // Cosine of the inner angle
	int pad1, pad2, pad3;
};

}

class Light : public Component
{
public:
	// Constructors
	explicit Light(const LightType t_type = LightType::Point);
	Light(const Light& t_light) = delete;
	Light(Light&& t_light) = default;

	// Destructor
	~Light() = default;

	// Operators
	Light& operator=(const Light& t_light) = delete;
	Light& operator=(Light&& t_light) = default;

	// Setters
	void setType(const LightType t_type);
	void setColor(const glm::vec3& t_color);
	void setIntensity(const float t_intensity);
	void setAttenuation(const glm::vec3& t_attenuation);
	void setRange(const float t_range);
	void setSpotAngle(const float t_spotAngle);
	void setSpotBlend(const float t_spotBlend);
	void resetAttenuation();

	// Getters
	LightType getType() const;
	const glm::vec3& getColor() const;
	float getIntensity() const;
	const glm::vec3& getAttenuation() const;
	float getRange() const;
	float getSpotAngle() const;
	float getSpotBlend() const;
private:
	// Attributes
	LightType m_type;
	glm::vec3 m_color;
	float m_intensity;
	glm::vec3 m_attenuation;
	float m_range;
	float m_spotAngle;
	float m_spotBlend;
	bool m_customAttenuation;
};

}
