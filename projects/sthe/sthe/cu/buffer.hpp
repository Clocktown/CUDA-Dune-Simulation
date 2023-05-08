#pragma once

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
	int getCount() const;
	int getStride() const;
	bool hasStorage() const;
	bool isMapped() const;

	template<typename T>
	const T* getData() const;

	template<typename T>
	T* getData();
private:
	// Attributes
	void* m_data;
	int m_count;
	int m_stride;
	bool m_isMapped;

	// Friend
	friend class GraphicsResource;
};

}
}

#include "buffer.inl"
