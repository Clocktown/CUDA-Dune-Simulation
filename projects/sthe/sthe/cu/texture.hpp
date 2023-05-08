 #pragma once

#include "array.hpp"
#include <cuda_runtime.h>

namespace sthe
{
namespace cu
{

class Texture
{
public:
	// Constructors
	Texture();
	explicit Texture(const Array& t_array, const cudaTextureDesc& t_descriptor =  {});
	explicit Texture(const cudaArray_t t_array, const cudaTextureDesc& t_descriptor = {});
	Texture(const Texture& t_texture) = delete;
	Texture(Texture&& t_texture) noexcept;

	// Destructor
	~Texture();

	// Operators
	Texture& operator=(const Texture& t_texture) = delete;
	Texture& operator=(Texture&& t_texture) noexcept;

	// Functionality
	void reinitialize(const Array& t_array, const cudaTextureDesc& t_descriptor = {});
	void reinitialize(const cudaArray_t t_array, const cudaTextureDesc& t_descriptor = {});
	void release();

	// Getters
	cudaTextureObject_t getHandle() const;
	bool hasResource() const;
private:
	// Attribute
	cudaTextureObject_t m_handle;
};

}
}
