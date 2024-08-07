#include "array.hpp"
#include "array3d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <sthe/gl/image.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>

namespace sthe
{
namespace cu
{

// Constructors
Array3D::Array3D() :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_depth{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{

}

Array3D::Array3D(const int t_width, const int t_height, const int t_depth, const cudaChannelFormatDesc& t_format, const unsigned int t_flags) :
	m_width{ t_width },
	m_height{ t_height },
	m_depth{ t_depth },
	m_format{ t_format },
	m_flags{ t_flags },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }

{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");

	if (m_width > 0)
	{
		const cudaExtent extent{ static_cast<size_t>(m_width), static_cast<size_t>(m_height), static_cast<size_t>(m_depth) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));
	}
	else
	{
		m_handle = nullptr;
	}
}

Array3D::Array3D(gl::Image& t_image, const unsigned int t_flags) :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_depth{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image.getHandle(), t_image.getTarget(), t_flags));
}

Array3D::Array3D(const GLuint t_image, const GLenum t_target, const unsigned int t_flags) :
	m_handle{ nullptr },
	m_width{ 0 },
	m_height{ 0 },
	m_depth{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_surface{ 0 },
	m_texture{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image, t_target, t_flags));
}

Array3D::Array3D(const Array3D& t_array3D) noexcept :
	m_width{ t_array3D.m_width },
	m_height{ t_array3D.m_height },
	m_depth{ t_array3D.m_depth },
	m_format{ t_array3D.m_format },
	m_flags{ t_array3D.m_flags },
	m_surface{ 0 },
	m_texture{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	if (t_array3D.hasStorage())
	{
		const cudaExtent extent{ static_cast<size_t>(m_width), static_cast<size_t>(m_height), static_cast<size_t>(m_depth) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));

		const cudaMemcpy3DParms parameter{ .srcArray{ t_array3D.m_handle },
								           .dstArray{ m_handle },
								           .extent{ make_cudaExtent(extent.width, static_cast<size_t>(std::max(m_height, 1)), static_cast<size_t>(std::max(m_depth, 1))) },
								           .kind{ cudaMemcpyDeviceToDevice } };

		CU_CHECK_ERROR(cudaMemcpy3D(&parameter));
	}
	else
	{
		m_handle = nullptr;
	}
}

Array3D::Array3D(Array3D&& t_array3D) noexcept :
	m_handle{ std::exchange(t_array3D.m_handle, nullptr) },
	m_width{ std::exchange(t_array3D.m_width, 0) },
	m_height{ std::exchange(t_array3D.m_height, 0) },
	m_depth{ std::exchange(t_array3D.m_depth, 0) },
	m_format{ std::exchange(t_array3D.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>()) },
	m_flags{ std::exchange(t_array3D.m_flags, cudaArrayDefault) },
	m_surface{ std::exchange(t_array3D.m_surface, 0) },
	m_texture{ std::exchange(t_array3D.m_texture, 0) },
	m_graphicsResource{ std::exchange(t_array3D.m_graphicsResource, nullptr) },
	m_isMapped{ std::exchange(t_array3D.m_isMapped, false) }
{

}

// Destructor
Array3D::~Array3D()
{
	release();
}

// Operators
Array3D& Array3D::operator=(const Array3D& t_array3D) noexcept
{
	if (this != &t_array3D)
	{
		reinitialize(t_array3D.m_width, t_array3D.m_height, t_array3D.m_depth, t_array3D.m_format, t_array3D.m_flags);

		if (t_array3D.hasStorage())
		{
			const cudaMemcpy3DParms parameter{ .srcArray{ t_array3D.m_handle },
								               .dstArray{ m_handle },
								               .extent{ make_cudaExtent(static_cast<size_t>(m_width), static_cast<size_t>(std::max(m_height, 1)), static_cast<size_t>(std::max(m_depth, 1))) },
								               .kind{ cudaMemcpyDeviceToDevice } };

			CU_CHECK_ERROR(cudaMemcpy3D(&parameter));
		}
	}

	return *this;
}

Array3D& Array3D::operator=(Array3D&& t_array3D) noexcept
{
	if (this != &t_array3D)
	{
		if (!m_isMapped)
		{
			CU_CHECK_ERROR(cudaFreeArray(m_handle));
		}

		m_handle = std::exchange(t_array3D.m_handle, nullptr);
		m_width = std::exchange(t_array3D.m_width, 0);
		m_height = std::exchange(t_array3D.m_height, 0);
		m_depth = std::exchange(t_array3D.m_depth, 0);
		m_format = std::exchange(t_array3D.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>());
		m_flags = std::exchange(t_array3D.m_flags, cudaArrayDefault);
		m_surface = std::exchange(t_array3D.m_surface, 0);
		m_texture = std::exchange(t_array3D.m_texture, 0);
		m_graphicsResource = std::exchange(t_array3D.m_graphicsResource, nullptr);
		m_isMapped = std::exchange(t_array3D.m_isMapped, false);
	}

	return *this;
}

// Functionality
void Array3D::reinitialize(const int t_width, const int t_height, const int t_depth, const cudaChannelFormatDesc& t_format, const unsigned int t_flags)
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");

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
	else if (m_width == t_width && m_height == t_height && m_depth == t_depth &&
			 m_format.x == t_format.x && m_format.y == t_format.y && m_format.z == t_format.z && m_format.w == t_format.w &&
			 m_format.f == t_format.f && m_flags == t_flags)
	{
		return;
	}

	m_width = t_width;
	m_height = t_height;
	m_depth = t_depth;
	m_format = t_format;
	m_flags = t_flags;
	
	if (m_width > 0)
	{
		const cudaExtent extent{ static_cast<size_t>(m_width), static_cast<size_t>(m_height), static_cast<size_t>(m_depth) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));
	}
	else
	{
		m_handle = nullptr;
	}
}

void Array3D::reinitialize(gl::Image& t_image, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image.getHandle(), t_image.getTarget(), t_flags));
}

void Array3D::reinitialize(const GLuint t_image, const GLenum t_target, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, t_image, t_target, t_flags));
}

cudaSurfaceObject_t Array3D::recreateSurface()
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ m_handle } } } };

	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_surface));
	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_surface, &resource));

	return m_surface;
}

cudaTextureObject_t Array3D::recreateTexture(const cudaTextureDesc& t_descriptor)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ m_handle } } } };

	CU_CHECK_ERROR(cudaDestroyTextureObject(m_texture));
	CU_CHECK_ERROR(cudaCreateTextureObject(&m_texture, &resource, &t_descriptor, nullptr));

	return m_texture;
}

void Array3D::release()
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
	m_depth = 0;
	m_surface = 0;
	m_texture = 0;
}

void Array3D::map(const int t_layer, const int t_mipLevel)
{
	STHE_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	STHE_ASSERT(!m_isMapped, "Array3D must be unmapped");

	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_graphicsResource));
	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&m_handle, m_graphicsResource, static_cast<unsigned int>(t_layer), static_cast<unsigned int>(t_mipLevel)));

	cudaExtent extent;
	CU_CHECK_ERROR(cudaArrayGetInfo(&m_format, &extent, &m_flags, m_handle));
	m_width = static_cast<int>(extent.width);
	m_height = static_cast<int>(extent.height);
	m_depth = static_cast<int>(extent.depth);
	m_isMapped = true;
}

void Array3D::unmap()
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
cudaArray_t Array3D::getHandle() const
{
	return m_handle;
}

int Array3D::getWidth() const
{
	return m_width;
}

int Array3D::getHeight() const
{
	return m_height;
}

int Array3D::getDepth() const
{
	return m_depth;
}

const cudaChannelFormatDesc& Array3D::getFormat() const
{
	return m_format;
}

unsigned int Array3D::getFlags() const
{
	return m_flags;
}

cudaSurfaceObject_t Array3D::getSurface() const
{
	return m_surface;
}

cudaTextureObject_t Array3D::getTexture() const
{
	return m_texture;
}

}
}
