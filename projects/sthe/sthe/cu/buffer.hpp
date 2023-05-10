#pragma once

#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <vector>

namespace sthe
{
namespace cu
{

class Buffer
{
public:
	// Constructors
	Buffer();
	Buffer(const int t_count, const int t_stride);
	explicit Buffer(gl::Buffer& t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	explicit Buffer(const GLuint t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
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
	void reinitialize(const int t_count, const int t_stride);
	void reinitialize(gl::Buffer& t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void reinitialize(const GLuint t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void reinterpret(const int t_stride);
	void release();
	void map(const int t_stride);
	void map();
	void unmap();

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
	int getCount() const;
	int getStride() const;
	bool hasStorage() const;

	template<typename T>
	const T* getData() const;

	template<typename T>
	T* getData();
private:
	// Attributes
	void* m_data;
	int m_count;
	int m_stride;

	cudaGraphicsResource_t m_graphicsResource;
	bool m_isMapped;
};

}
}

#include "buffer.inl"
