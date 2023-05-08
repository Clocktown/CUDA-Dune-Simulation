#include "material.hpp"
#include "custom_material.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/config/binding.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture2d.hpp>
#include <glm/glm.hpp>
#include <memory>

namespace sthe
{

// Constructors
Material::Material(const glm::vec3& t_diffuseColor, const glm::vec3& t_specularColor, const float t_specularIntensity, const float t_shininess, const float t_opacity) :
    Material{ nullptr, t_diffuseColor, t_specularColor, t_specularIntensity, t_shininess, t_opacity }
{

}

Material::Material(const std::shared_ptr<gl::Program>& t_program, const glm::vec3& t_diffuseColor, const glm::vec3& t_specularColor, const float t_specularIntensity, const float t_shininess, const float t_opacity) :
    CustomMaterial{ t_program },
    m_diffuseColor{ t_diffuseColor },
    m_specularColor{ t_specularColor },
    m_specularIntensity{ t_specularIntensity },
    m_shininess{ t_shininess },
    m_opacity{ t_opacity }
{

}

// Setters
void Material::setDiffuseColor(const glm::vec3& t_diffuseColor)
{
    m_diffuseColor = t_diffuseColor;
}

void Material::setSpecularColor(const glm::vec3& t_specularColor)
{
    m_specularColor = t_specularColor;
}

void Material::setSpecularIntensity(const float t_specularIntensity)
{
    m_specularIntensity = t_specularIntensity;
}

void Material::setShininess(const float t_shininess)
{
    m_shininess = t_shininess;
}

void Material::setOpacity(const float t_opacity)
{
    m_opacity = t_opacity;
}

void Material::setDiffuseMap(const std::shared_ptr<gl::Texture2D>& t_diffuseMap)
{
    setTexture(STHE_TEXTURE_UNIT_MATERIAL_DIFFUSE, t_diffuseMap);
}

void Material::setNormalMap(const std::shared_ptr<gl::Texture2D>& t_normalMap)
{
    setTexture(STHE_TEXTURE_UNIT_MATERIAL_NORMAL, t_normalMap);
}

// Getters
const glm::vec3& Material::getDiffuseColor() const
{
    return m_diffuseColor;
}

const glm::vec3& Material::getSpecularColor() const
{
    return m_specularColor;
}

float Material::getSpecularIntensity() const
{
    return m_specularIntensity;
}

float Material::getShininess() const
{
    return m_shininess;
}

float Material::getOpacity() const
{
    return m_opacity;
}

std::shared_ptr<gl::Texture2D> Material::getDiffuseMap() const
{
    const std::shared_ptr<gl::Texture> diffuseMap{ getTexture(STHE_TEXTURE_UNIT_MATERIAL_DIFFUSE) };

    STHE_ASSERT(diffuseMap == nullptr || diffuseMap->getTarget() == GL_TEXTURE_2D, "Diffuse map must target GL_TEXTURE_2D");

    return std::static_pointer_cast<gl::Texture2D>(diffuseMap);
}

std::shared_ptr<gl::Texture2D> Material::getNormalMap() const
{
    const std::shared_ptr<gl::Texture> normalMap{ getTexture(STHE_TEXTURE_UNIT_MATERIAL_NORMAL) };

    STHE_ASSERT(normalMap == nullptr || normalMap->getTarget() == GL_TEXTURE_2D, "Normal map must target GL_TEXTURE_2D");

    return std::static_pointer_cast<gl::Texture2D>(normalMap);
}

bool Material::hasDiffuseMap() const
{
    return hasTexture(STHE_TEXTURE_UNIT_MATERIAL_DIFFUSE);
}

bool Material::hasNormalMap() const
{
    return hasTexture(STHE_TEXTURE_UNIT_MATERIAL_NORMAL);
}


}
