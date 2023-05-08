#pragma once

#include <sthe/gl/texture2d.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace sthe
{

namespace uniform
{

struct TerrainLayer
{
	glm::vec3 diffuseColor;
	float specularIntensity;
	glm::vec3 specularColor;
	float shininess;
	int hasDiffuseMap;
	int pad1, pad2, pad3;
};

}

class TerrainLayer
{
public:
	// Constructors
	explicit TerrainLayer(const glm::vec3& t_diffuseColor = glm::vec3{ 1.0f }, const glm::vec3& t_specularColor = glm::vec3{ 1.0f }, const float t_specularIntensity = 0.0f, const float t_shininess = 80.0f);
	TerrainLayer(const TerrainLayer& t_terrainLayer) = default;
	TerrainLayer(TerrainLayer&& t_terrainLayer) = default;

	// Destructor
	~TerrainLayer() = default;

	// Operators
	TerrainLayer& operator=(const TerrainLayer& t_terrainLayer) = default;
	TerrainLayer& operator=(TerrainLayer&& t_terrainLayer) = default;

	// Setters
	void setDiffuseColor(const glm::vec3& t_diffuseColor);
	void setSpecularColor(const glm::vec3& t_specularColor);
	void setSpecularIntensity(const float t_specularIntensity);
	void setShininess(const float t_shininess);
	void setDiffuseMap(const std::shared_ptr<sthe::gl::Texture2D>& t_diffuseMap);

	// Getters
	const glm::vec3& getDiffuseColor() const;
	const glm::vec3& getSpecularColor() const;
	float getSpecularIntensity() const;
	float getShininess() const;
	const std::shared_ptr<sthe::gl::Texture2D>& getDiffuseMap() const;
	bool hasDiffuseMap() const;
private:
	// Attributes
	glm::vec3 m_diffuseColor;
	glm::vec3 m_specularColor;
	float m_specularIntensity;
	float m_shininess;
	std::shared_ptr<sthe::gl::Texture2D> m_diffuseMap;
};

}
