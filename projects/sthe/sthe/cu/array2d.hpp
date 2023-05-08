#pragma once

#include "array.hpp"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace sthe
{
namespace cu
{

class Array2D : public Array
{
public:
	// Constructors
	Array2D();
	Array2D(const int t_width, const int t_height, const cudaChannelFormatDesc& t_format, const unsigned int t_flags = cudaArrayDefault);
	Array2D(const std::string& t_file, const unsigned int t_flags = cudaArrayDefault);
	Array2D(const Array2D& t_array2D) noexcept;
	Array2D(Array2D&& t_array2D) noexcept;

	// Destructor
	~Array2D();
	
	// Operators
	Array2D& operator=(const Array2D& t_array2D) noexcept;
	Array2D& operator=(Array2D&& t_array2D) noexcept;

	// Functionality
	void reinitialize(const int t_width, const int t_height, const cudaChannelFormatDesc& t_format, const unsigned int t_flags = cudaArrayDefault);
	void release() override;

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_width, const int t_height);

	template<typename T>
	void upload(const T* const t_source, const int t_width, const int t_height);

	template<typename T>
	void upload(const T* const t_source, const int t_x, const int t_y, const int t_width, const int t_height);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_width, const int t_height);
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_width, const int t_height) const;
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_width, const int t_height) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_width, const int t_height) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_x, const int t_y, const int t_width, const int t_height) const;

	// Getters
	cudaArray_t getHandle() const override;
	int getWidth() const override;
	int getHeight() const override;
	const cudaChannelFormatDesc& getFormat() const override;
	unsigned int getFlags() const override;
	bool isMapped() const override;
private:
	// Attributes
	cudaArray_t m_handle;
	int m_width;
	int m_height;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;
	bool m_isMapped;

	// Friend
	friend class GraphicsResource;
};

}
}

#include "array2D.inl"
