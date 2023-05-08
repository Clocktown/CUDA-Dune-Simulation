#pragma once

#include "custom_material.hpp"
#include <sthe/gl/program.hpp>
#include <sthe/gl/texture2d.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <map>

namespace sthe
{

namespace uniform
{

struct Material
{
	glm::vec3 diffuseColor;
	float opacity;
	glm::vec3 specularColor;
	float specularIntensity;
	float shininess;
	int hasDiffuseMap;
	int hasNormalMap;
	int pad;
};

}

class Material : public CustomMaterial
{
public:
	// Constructors
	explicit Material(const glm::vec3& t_diffuseColor = glm::vec3{ 1.0f }, const glm::vec3& t_specularColor = glm::vec3{ 1.0f }, const float t_specularIntensity = 0.0f, const float t_shininess = 80.0f, const float t_opacity = 1.0f);
	explicit Material(const std::shared_ptr<gl::Program>& t_program, const glm::vec3& t_diffuseColor = glm::vec3{ 1.0f }, const glm::vec3& t_specularColor = glm::vec3{ 1.0f }, const float t_specularIntensity = 0.0f, const float t_shininess = 80.0f, const float t_opacity = 1.0f);
	Material(const Material& t_material) = default;
	Material(Material&& t_material) = default;

	// Destructor
	~Material() = default;

	// Operators
	Material& operator=(const Material& t_material) = default;
	Material& operator=(Material&& t_material) = default;

	// Setters
	void setDiffuseColor(const glm::vec3& t_diffuseColor);
	void setSpecularColor(const glm::vec3& t_specularColor);
	void setSpecularIntensity(const float t_specularIntensity);
	void setShininess(const float t_shininess);
	void setOpacity(const float t_opacity);
	void setDiffuseMap(const std::shared_ptr<gl::Texture2D>& t_diffuseMap);
	void setNormalMap(const std::shared_ptr<gl::Texture2D>& t_normalMap);
	
	// Getters
	const glm::vec3& getDiffuseColor() const;
	const glm::vec3& getSpecularColor() const;
	float getSpecularIntensity() const;
	float getShininess() const;
	float getOpacity() const;
	std::shared_ptr<gl::Texture2D> getDiffuseMap() const;
	std::shared_ptr<gl::Texture2D> getNormalMap() const;
	bool hasDiffuseMap() const;
	bool hasNormalMap() const;
private:
	// Attributes
	glm::vec3 m_diffuseColor;
	glm::vec3 m_specularColor;
	float m_specularIntensity;
	float m_shininess;
	float m_opacity;
};

}
