#pragma once

#include "array.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace sthe
{
namespace cu
{

class Array3D : public Array
{
public:
	// Constructors
	Array3D();
	Array3D(const int t_width, const int t_height, const int t_depth, const cudaChannelFormatDesc& t_format, const unsigned int t_flags = cudaArrayDefault);
	Array3D(const Array3D& t_array3D) noexcept;
	Array3D(Array3D&& t_array3D) noexcept;

	// Destructor
	~Array3D();
	
	// Operators
	Array3D& operator=(const Array3D& t_array3D) noexcept;
	Array3D& operator=(Array3D&& t_array3D) noexcept;

	// Functionality
	void reinitialize(const int t_width, const int t_height, const int t_depth, const cudaChannelFormatDesc& t_format, const unsigned int t_flags = cudaArrayDefault);
	void release() override;

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_width, const int t_height, const int t_depth);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth);
	
	template<typename T>
	void upload(const T* const t_source, const int t_width, const int t_height, const int t_depth);
	
	template<typename T>
	void upload(const T* const t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth);
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_width, const int t_height, const int t_depth) const;
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_width, const int t_height, const int t_depth) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth) const;

	// Getters
	cudaArray_t getHandle() const override;
	int getWidth() const;
	int getHeight() const;
	int getDepth() const;
	const cudaChannelFormatDesc& getFormat() const override;
	unsigned int getFlags() const override;
	bool isMapped() const override;
private:
	// Attributes
	cudaArray_t m_handle;
	int m_width;
	int m_height;
	int m_depth;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;
	bool m_isMapped;

	// Friend
	friend class GraphicsResource;
};

}
}

#include "array3D.inl"
