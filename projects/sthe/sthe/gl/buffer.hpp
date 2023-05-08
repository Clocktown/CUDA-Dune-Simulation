#pragma once

#include "handle.hpp"
#include <glad/glad.h>
#include <vector>

namespace sthe
{
namespace gl
{

class Buffer : public Handle
{
public:
	// Constructors
	Buffer();
	Buffer(const int t_count, const int t_stride);
	Buffer(const Buffer& t_buffer) noexcept;
	Buffer(Buffer&& t_buffer) noexcept;

	template<typename T>
	explicit Buffer(const std::vector<T>& t_source);

	// Destructor
	~Buffer();

	// Operators
	Buffer& operator=(const Buffer& t_buffer) noexcept;
	Buffer& operator=(Buffer&& t_buffer) noexcept;

	// Functionality
	void bind(const GLenum t_target) const;
	void bind(const GLenum t_target, const int t_location) const;
	void bind(const GLenum t_target, const int t_location, const int t_count) const;
	void bind(const GLenum t_target, const int t_location, const int t_offset, const int t_count) const;
	void reinitialize(const int t_count, const int t_stride);
	void reinterpret(const int t_stride);
	void release();

	template<typename T>
	void reinitialize(const std::vector<T>& t_source);

	template<typename T>
	void upload(const std::vector<T>& t_source);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_count);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_offset, const int t_count);

	template<typename T>
	void upload(const T* const t_source, const int t_count);

	template<typename T>
	void upload(const T* const t_source, const int t_offset, const int t_count);

	template<typename T>
	void download(std::vector<T>& t_destination) const;

	template<typename T>
	void download(std::vector<T>& t_destination, const int t_count) const;

	template<typename T>
	void download(std::vector<T>& t_destination, const int t_offset, const int t_count) const;

	template<typename T>
	void download(T* const t_destination, const int t_count) const;

	template<typename T>
	void download(T* const t_destination, const int t_offset, const int t_count) const;

	// Getters
	GLuint getHandle() const override;
	int getCount() const;
	int getStride() const;
	bool hasStorage() const;
private:
	// Attributes
	GLuint m_handle;
	int m_count;
	int m_stride;
};

}
}

#include "buffer.inl"
