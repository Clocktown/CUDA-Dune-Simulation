#pragma once

#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture.hpp>
#include <memory>
#include <map>

namespace sthe
{

class CustomMaterial
{
public:
	// Constructors
	explicit CustomMaterial(const std::shared_ptr<gl::Program>& t_program = nullptr);
	CustomMaterial(const CustomMaterial& t_customMaterial) = default;
	CustomMaterial(CustomMaterial&& t_customMaterial) = default;

	// Destructor
	virtual ~CustomMaterial() = default;

	// Operators
	CustomMaterial& operator=(const CustomMaterial& t_customMaterial) = default;
	CustomMaterial& operator=(CustomMaterial&& t_customMaterial) = default;

	// Functionality
	void use() const;
	void bind() const;

	// Setters
	void setProgram(const std::shared_ptr<gl::Program>& t_program);
	void setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer);
	void setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer, const int t_count);
	void setBuffer(const GLenum t_target, const int t_location, const std::shared_ptr<gl::Buffer>& t_buffer, const int t_offset, const int t_count);
	void setTexture(const int t_unit, const std::shared_ptr<gl::Texture>& t_texture);

	// Getters
	const std::shared_ptr<gl::Program>& getProgram() const;
	std::shared_ptr<gl::Buffer> getBuffer(const GLenum t_target, const int t_location) const;
	std::shared_ptr<gl::Texture> getTexture(const int t_unit) const;
	bool hasProgram() const;
	bool hasBuffer(const GLenum t_target, const int t_location) const;
	bool hasTexture(const int t_unit) const;
private:
	struct BufferDescriptor
	{
		std::shared_ptr<gl::Buffer> buffer;
		int offset;
		int count;
	};

	// Attributes
	std::shared_ptr<gl::Program> m_program;
	std::map<std::pair<GLenum, int>, BufferDescriptor> m_buffers;
	std::map<int, std::shared_ptr<gl::Texture>> m_textures;
};

}
