#include "texture.hpp"
#include "array.hpp"
#include <sthe/config/debug.hpp>
#include <cuda_runtime.h>
#include <utility>

namespace sthe
{
namespace cu
{

// Constructor
Texture::Texture() :
	m_handle{ 0 }
{

}

Texture::Texture(const Array& t_array, const cudaTextureDesc& t_descriptor) :
	Texture{ t_array.getHandle(), t_descriptor }
{

}

Texture::Texture(const cudaArray_t t_array, const cudaTextureDesc& t_descriptor)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
							         .res{ .array{ .array{ t_array } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &t_descriptor, nullptr));
}

Texture::Texture(Texture&& t_texture) noexcept :
	m_handle{ std::exchange(t_texture.m_handle, 0) }
{

}

// Destructor
Texture::~Texture()
{
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
}

// Operator
Texture& Texture::operator=(Texture&& t_texture) noexcept
{
	if (this != &t_texture)
	{
		CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));

		m_handle = std::exchange(t_texture.m_handle, 0);
	}

	return *this;
}

// Functionality
void Texture::reinitialize(const Array& t_array, const cudaTextureDesc& t_descriptor)
{
	reinitialize(t_array.getHandle(), t_descriptor);
}

void Texture::reinitialize(const cudaArray_t t_array, const cudaTextureDesc& t_descriptor)
{
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));

	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
									 .res{ .array{ .array{ t_array } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &t_descriptor, nullptr));
}

void Texture::release()
{
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
}

// Getters
cudaTextureObject_t Texture::getHandle() const
{
	return m_handle;
}

bool Texture::hasResource() const
{
	return m_handle == 0;
}

}
}
