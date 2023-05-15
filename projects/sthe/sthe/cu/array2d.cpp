#include "array.hpp"
#include "array2d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <sthe/gl/image.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>
#include <memory>
#include <string>

namespace sthe
{
namespace cu
{

// Constructors
Array2D::Array2D() :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	
}

Array2D::Array2D(const int t_width, const int t_height, const cudaChannelFormatDesc& t_format, const unsigned int t_flags) :
	m_width{ t_width },
	m_height{ t_height },
	m_format{ t_format },
	m_flags{ t_flags },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");

	if (m_width > 0)
	{
		CU_CHECK_ERROR(cudaMallocArray(&m_handle, &m_format, static_cast<size_t>(m_width), static_cast<size_t>(m_height), m_flags));
	}
	else
	{
		m_handle = nullptr;
	}
}

Array2D::Array2D(gl::Image& t_image, const unsigned int t_flags) :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image.getHandle(), t_image.getTarget(), t_flags));
}

Array2D::Array2D(const GLuint t_image, const GLenum t_target, const unsigned int t_flags) :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image, t_target, t_flags));
}

Array2D::Array2D(const std::string& t_file, const unsigned int t_flags) :
	m_flags{ t_flags },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	const std::shared_ptr<unsigned char> source{ readImage2D(t_file, m_width, m_height, m_format) };

	const size_t width{ static_cast<size_t>(m_width) };
	const size_t height{ static_cast<size_t>(m_height) };
	const size_t pitch{ width * static_cast<size_t>(getStride()) };
	CU_CHECK_ERROR(cudaMallocArray(&m_handle, &m_format, width, height, m_flags));
	CU_CHECK_ERROR(cudaMemcpy2DToArray(m_handle, 0, 0, source.get(), pitch, width, height, cudaMemcpyHostToDevice));
}

Array2D::Array2D(const Array2D& t_array2D) noexcept :
	m_width{ t_array2D.m_width },
	m_height{ t_array2D.m_height },
	m_format{ t_array2D.m_format },
	m_flags{ t_array2D.m_flags },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	if (t_array2D.hasStorage())
	{
		const size_t width{ static_cast<size_t>(m_width) };
		const size_t height{ static_cast<size_t>(std::max(m_height, 1)) };
		CU_CHECK_ERROR(cudaMallocArray(&m_handle, &m_format, width, height, m_flags));
		CU_CHECK_ERROR(cudaMemcpy2DArrayToArray(m_handle, 0, 0, t_array2D.m_handle, 0, 0, width * static_cast<size_t>(getStride()), height, cudaMemcpyDeviceToDevice));
	}
	else
	{
		m_handle = nullptr;
	}
}

Array2D::Array2D(Array2D&& t_array2D) noexcept :
	m_handle{ std::exchange(t_array2D.m_handle, nullptr) },
	m_width{ std::exchange(t_array2D.m_width, 0) },
	m_height{ std::exchange(t_array2D.m_height, 0) },
	m_format{ std::exchange(t_array2D.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>()) },
	m_flags{ std::exchange(t_array2D.m_flags, cudaArrayDefault) },
	m_surface{ std::exchange(t_array2D.m_surface, 0) },
	m_texture{ std::exchange(t_array2D.m_texture, 0) },
	m_graphicsResource{ std::exchange(t_array2D.m_graphicsResource, nullptr) },
	m_isMapped{ std::exchange(t_array2D.m_isMapped, false) }
{

}

// Destructor
Array2D::~Array2D()
{
	release();
}

// Operators
Array2D& Array2D::operator=(const Array2D& t_array2D) noexcept
{
	if (this != &t_array2D)
	{
		reinitialize(t_array2D.m_width, t_array2D.m_height, t_array2D.m_format, t_array2D.m_flags);

		if (t_array2D.hasStorage())
		{
			const size_t height{ static_cast<size_t>(std::max(m_height, 1)) };
			CU_CHECK_ERROR(cudaMemcpy2DArrayToArray(m_handle, 0, 0, t_array2D.m_handle, 0, 0, static_cast<size_t>(m_width) * static_cast<size_t>(getStride()), height, cudaMemcpyDeviceToDevice));
		}
	}

	return *this;
}

Array2D& Array2D::operator=(Array2D&& t_array2D) noexcept
{
	if (this != &t_array2D)
	{
		if (!m_isMapped)
		{
			CU_CHECK_ERROR(cudaFreeArray(m_handle));
		}

		m_handle = std::exchange(t_array2D.m_handle, nullptr);
		m_width = std::exchange(t_array2D.m_width, 0);
		m_height = std::exchange(t_array2D.m_height, 0);
		m_format = std::exchange(t_array2D.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>());
		m_flags = std::exchange(t_array2D.m_flags, cudaArrayDefault);
		m_surface = std::exchange(t_array2D.m_surface, 0);
		m_texture = std::exchange(t_array2D.m_texture, 0);
		m_graphicsResource = std::exchange(t_array2D.m_graphicsResource, nullptr);
		m_isMapped = std::exchange(t_array2D.m_isMapped, false);
	}

	return *this;
}

// Functionality
void Array2D::reinitialize(const int t_width, const int t_height, const cudaChannelFormatDesc& t_format, const unsigned int t_flags)
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");

	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_surface));
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_texture));
	m_surface = 0;
	m_texture = 0;

	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else if (m_width == t_width && m_height == t_height &&
		     m_format.x == t_format.x && m_format.y == t_format.y && m_format.z == t_format.z && m_format.w == t_format.w &&
		     m_format.f == t_format.f && m_flags == t_flags)
	{
		return;
	}

	m_width = t_width;
	m_height = t_height;
	m_format = t_format;
	m_flags = t_flags;

	if (m_width > 0)
	{
		CU_CHECK_ERROR(cudaMallocArray(&m_handle, &m_format, static_cast<size_t>(m_width), static_cast<size_t>(m_height), m_flags));
	}
	else
	{
		m_handle = nullptr;
	}
}

void Array2D::reinitialize(gl::Image& t_image, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image.getHandle(), t_image.getTarget(), t_flags));
}

void Array2D::reinitialize(const GLuint t_image, const GLenum t_target, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image, t_target, t_flags));
}

cudaSurfaceObject_t Array2D::recreateSurface()
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ m_handle } } } };

	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_surface));
	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_surface, &resource));

	return m_surface;
}

cudaTextureObject_t Array2D::recreateTexture(const cudaTextureDesc& t_descriptor)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ m_handle } } } };

	CU_CHECK_ERROR(cudaDestroyTextureObject(m_texture));
	CU_CHECK_ERROR(cudaCreateTextureObject(&m_texture, &resource, &t_descriptor, nullptr));

	return m_texture;
}

void Array2D::release()
{
	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_surface));
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_texture));

	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else
	{
		CU_CHECK_ERROR(cudaFreeArray(m_handle));
	}

	m_handle = nullptr;
	m_width = 0;
	m_height = 0;
	m_surface = 0;
	m_texture = 0;
}

void Array2D::map(const int t_layer, const int t_mipLevel)
{
	STHE_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	STHE_ASSERT(!m_isMapped, "Array2D must be unmapped");

	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_graphicsResource));
	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&m_handle, m_graphicsResource, static_cast<unsigned int>(t_layer), static_cast<unsigned int>(t_mipLevel)));

	cudaExtent extent;
	CU_CHECK_ERROR(cudaArrayGetInfo(&m_format, &extent, &m_flags, m_handle));
	m_width = static_cast<int>(extent.width);
	m_height = static_cast<int>(extent.height);
	m_isMapped = true;
}

void Array2D::unmap()
{
	STHE_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	STHE_ASSERT(m_isMapped, "Array2D must be mapped");

	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_surface));
	CU_CHECK_ERROR(cudaDestroyTextureObject(m_texture));
	m_surface = 0;
	m_texture = 0;

	CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
	m_isMapped = false;
}

// Getters
cudaArray_t Array2D::getHandle() const
{
	return m_handle;
}

int Array2D::getWidth() const
{
	return m_width;
}

int Array2D::getHeight() const
{
	return m_height;
}

const cudaChannelFormatDesc& Array2D::getFormat() const
{
	return m_format;
}

unsigned int Array2D::getFlags() const
{
	return m_flags;
}

cudaSurfaceObject_t Array2D::getSurface() const
{
	return m_surface;
}

cudaTextureObject_t Array2D::getTexture() const
{
	return m_texture;
}


}
}
