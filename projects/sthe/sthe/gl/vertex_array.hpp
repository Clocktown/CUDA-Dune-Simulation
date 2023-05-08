#pragma once

#include "handle.hpp"
#include "buffer.hpp"
#include <glad/glad.h>

namespace sthe
{
namespace gl
{

class VertexArray : public Handle
{
public:
	// Constructors
	VertexArray();
	VertexArray(const VertexArray& t_vertexArray) = delete;
	VertexArray(VertexArray&& t_vertexArray) noexcept;

	// Destructor
	~VertexArray();

	// Operators
	VertexArray& operator=(const VertexArray& t_vertexArray) = delete;
	VertexArray& operator=(VertexArray&& t_vertexArray) noexcept;

	// Functionality
	void bind() const;
	void attachIndexBuffer(const Buffer& t_indexBuffer);
	void attachIndexBuffer(const GLuint t_indexBuffer);
	void attachVertexBuffer(const int t_location, const Buffer& t_vertexBuffer, const int t_offset = 0);
	void attachVertexBuffer(const int t_location, const GLuint t_vertexBuffer, const int t_stride);
	void attachVertexBuffer(const int t_location, const GLuint t_vertexBuffer, const long long int t_offset, const int t_stride);
	void detachIndexBuffer();
	void detachVertexBuffer(const int t_location);

	// Setters
	void setVertexAttributeFormat(const int t_location, const int t_count, const GLenum t_type, const bool t_isNormalized = false, const int t_relativeStride = 0);
	void setVertexAttributeDivisor(const int t_location, const int t_divisor);
	void enableVertexAttribute(const int t_location);
	void disableVertexAttribute(const int t_location);
	
	// Getter
	GLuint getHandle() const override;
private:
	// Attribute
	GLuint m_handle;
};

}
}
