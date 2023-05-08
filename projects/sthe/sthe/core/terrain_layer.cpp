#include "terrain_layer.hpp"
#include <sthe/gl/texture2d.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace sthe
{

// Constructor
TerrainLayer::TerrainLayer(const glm::vec3& t_diffuseColor, const glm::vec3& t_specularColor, const float t_specularIntensity, const float t_shininess) : 
    m_diffuseColor{ t_diffuseColor },
    m_specularColor{ t_specularColor },
    m_specularIntensity{ t_specularIntensity },
    m_shininess{ t_shininess },
    m_diffuseMap{ nullptr }
{

}

// Setters
void TerrainLayer::setDiffuseColor(const glm::vec3& t_diffuseColor)
{
    m_diffuseColor = t_diffuseColor;
}

void TerrainLayer::setSpecularColor(const glm::vec3& t_specularColor)
{
    m_specularColor = t_specularColor;
}

void TerrainLayer::setSpecularIntensity(const float t_specularIntensity)
{
    m_specularIntensity = t_specularIntensity;
}

void TerrainLayer::setShininess(const float t_shininess)
{
    m_shininess = t_shininess;
}

void TerrainLayer::setDiffuseMap(const std::shared_ptr<sthe::gl::Texture2D>& t_diffuseMap)
{
    m_diffuseMap = t_diffuseMap;
}

// Getters
const glm::vec3& TerrainLayer::getDiffuseColor() const
{
    return m_diffuseColor;
}

const glm::vec3& TerrainLayer::getSpecularColor() const
{
    return m_specularColor;
}

float TerrainLayer::getSpecularIntensity() const
{
    return m_specularIntensity;
}

float TerrainLayer::getShininess() const
{
    return m_shininess;
}

const std::shared_ptr<sthe::gl::Texture2D>& TerrainLayer::getDiffuseMap() const
{
    return m_diffuseMap;
}

bool TerrainLayer::hasDiffuseMap() const
{
    return m_diffuseMap != nullptr;
}

}
