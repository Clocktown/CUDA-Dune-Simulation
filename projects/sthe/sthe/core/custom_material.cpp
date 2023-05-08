#include "custom_material.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture.hpp>
#include <sthe/gl/texture2d.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

#define STHE_PROGRAM_MATERIAL_BUFFER_FULL_RANGE 0

namespace sthe
{

// Constructors
CustomMaterial::CustomMaterial(const std::shared_ptr<gl::Program>& t_program) :
    m_program{ t_program }
{

}

// Functionality
void CustomMaterial::use() const
{
    if (hasProgram())
    {
        m_program->use();
    }

    bind();
}

void CustomMaterial::bind() const
{
    for (const auto& [location, descriptor] : m_buffers)
    {
        if (descriptor.count == STHE_PROGRAM_MATERIAL_BUFFER_FULL_RANGE)
        {
            descriptor.buffer->bind(location.first, location.second);
        }
        else
        {
            descriptor.buffer->bind(location.first, location.second, descriptor.offset, descriptor.count);
        }
    }

    for (const auto& [unit, texture] : m_textures)
    {
        texture->bind(unit);
    }
}

// Setters
void CustomMaterial::setProgram(const std::shared_ptr<gl::Program>& t_program)
{
    m_program = t_program;
}

void CustomMaterial::setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer)
{
    setBuffer(t_target, t_location, t_buffer, 0, STHE_PROGRAM_MATERIAL_BUFFER_FULL_RANGE);
}

void CustomMaterial::setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer, const int t_count)
{
    setBuffer(t_target, t_location, t_buffer, 0, t_count);
}

void CustomMaterial::setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer, const int t_offset, const int t_count)
{
    const std::pair<GLenum, int> location{ t_target, t_location };

    if (t_buffer != nullptr)
    {
        m_buffers[location] = BufferDescriptor{ t_buffer, t_offset, t_count };
    }
    else
    {
        m_buffers.erase(location);
    }
}

void CustomMaterial::setTexture(const int t_unit, const std::shared_ptr<gl::Texture>& t_texture)
{
    if (t_texture != nullptr)
    {
        m_textures[t_unit] = t_texture;
    }
    else
    {
        m_textures.erase(t_unit);
    }
}

// Getters
const std::shared_ptr<gl::Program>& CustomMaterial::getProgram() const
{
    return m_program;
}

std::shared_ptr<gl::Buffer> CustomMaterial::getBuffer(const GLenum t_target, const int t_location) const
{
    const auto iterator{ m_buffers.find(std::pair<GLenum, int>{ t_target, t_location }) };

    if (iterator != m_buffers.end())
    {
        return iterator->second.buffer;
    }

    return nullptr;
}

std::shared_ptr<gl::Texture> CustomMaterial::getTexture(const int t_unit) const
{
    const auto iterator{ m_textures.find(t_unit) };

    if (iterator != m_textures.end())
    {
        return iterator->second;
    }

    return nullptr;
}

bool CustomMaterial::hasProgram() const
{
    return m_program != nullptr;
}

bool CustomMaterial::hasBuffer(const GLenum t_target, const int t_location) const
{
    return m_buffers.contains(std::pair<GLenum, int>{ t_target, t_location });
}

bool CustomMaterial::hasTexture(const int t_unit) const
{
    return m_textures.contains(t_unit);
}

}
