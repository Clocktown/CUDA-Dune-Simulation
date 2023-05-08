#include "graphics_resource.hpp"
#include "buffer.hpp"
#include "array2d.hpp"
#include "array3d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/image.hpp>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>

namespace sthe
{
namespace cu
{

// Constructors
GraphicsResource::GraphicsResource() :
    m_handle{ nullptr },
	m_isMapped{ false }
{
	
}

GraphicsResource::GraphicsResource(GraphicsResource&& t_graphicsResource) noexcept :
    m_handle{ std::exchange(t_graphicsResource.m_handle, nullptr) },
	m_isMapped{ std::exchange(t_graphicsResource.m_isMapped, false) }
{

}

// Destructor
GraphicsResource::~GraphicsResource()
{
	unmap();
	unregister();
}

// Operator
GraphicsResource& GraphicsResource::operator=(GraphicsResource&& t_graphicsResource) noexcept
{
	if (this != &t_graphicsResource)
	{
		unmap();
		unregister();

		m_handle = std::exchange(t_graphicsResource.m_handle, nullptr);
		m_isMapped = std::exchange(t_graphicsResource.m_isMapped, false);
	}

	return *this;
}

void GraphicsResource::registerBuffer(gl::Buffer& t_buffer, const unsigned int t_flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_handle, t_buffer.getHandle(), t_flags));
}

void GraphicsResource::registerBuffer(const GLuint t_buffer, const unsigned int t_flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_handle, t_buffer, t_flags));
}

void GraphicsResource::registerImage(gl::Image& t_image, const unsigned int t_flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, t_image.getHandle(), t_image.getTarget(), t_flags));
}

void GraphicsResource::registerImage(const GLuint t_image, const GLenum t_target, const unsigned int t_flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, t_image, t_target, t_flags));
}

void GraphicsResource::unregister()
{
	if (isRegistered())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
		m_handle = 0;
	}
}

void GraphicsResource::mapBuffer(Buffer& t_buffer)
{
	mapBuffer(t_buffer, t_buffer.getStride());
}

void GraphicsResource::mapBuffer(Buffer& t_buffer, const int t_stride)
{
	STHE_ASSERT(t_stride > 0, "Stride must be greater than 0");

	if (t_buffer.m_data != 0 && !t_buffer.m_isMapped)
	{
		CU_CHECK_ERROR(cudaFree(t_buffer.m_data));
	}

	map();

	size_t size;
	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&t_buffer.m_data, &size, m_handle));
	const size_t stride{ static_cast<size_t>(t_stride) };

	STHE_ASSERT(size % stride == 0, "Size must be reinterpretable with stride");
	
	t_buffer.m_count = static_cast<int>(size / stride);
	t_buffer.m_stride = t_stride;
	t_buffer.m_isMapped = true;
}

void GraphicsResource::mapImage(Array2D& t_array2D, const int t_layer, const int t_mipLevel)
{
	if (t_array2D.m_handle != 0 && !t_array2D.m_isMapped)
	{
		CU_CHECK_ERROR(cudaFreeArray(t_array2D.m_handle));
	}

	mapImage(t_array2D.m_handle, t_layer, t_mipLevel);

	cudaExtent extent;
	CU_CHECK_ERROR(cudaArrayGetInfo(&t_array2D.m_format, &extent, &t_array2D.m_flags, t_array2D.m_handle));
	t_array2D.m_width = static_cast<int>(extent.width);
	t_array2D.m_height = static_cast<int>(extent.height);
	t_array2D.m_isMapped = true;
}

void GraphicsResource::mapImage(Array3D& t_array3D, const int t_layer, const int t_mipLevel)
{
	if (t_array3D.m_handle != 0 && !t_array3D.m_isMapped)
	{
		CU_CHECK_ERROR(cudaFreeArray(t_array3D.m_handle));
	}

	mapImage(t_array3D.m_handle, t_layer, t_mipLevel);

	cudaExtent extent;
	CU_CHECK_ERROR(cudaArrayGetInfo(&t_array3D.m_format, &extent, &t_array3D.m_flags, t_array3D.m_handle));
	t_array3D.m_width = static_cast<int>(extent.width);
	t_array3D.m_height = static_cast<int>(extent.height);
	t_array3D.m_depth = static_cast<int>(extent.depth);
	t_array3D.m_isMapped = true;
}

void GraphicsResource::mapImage(cudaArray_t& t_image, const int t_layer, const int t_mipLevel)
{
	STHE_ASSERT(t_layer >= 0, "Layer must be greater than or equal to 0");
	STHE_ASSERT(t_mipLevel >= 0, "Level must be greater than or equal to 0");

	map();
	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&t_image, m_handle, static_cast<unsigned int>(t_layer), static_cast<unsigned int>(t_mipLevel)));
}

void GraphicsResource::map()
{
	if (!m_isMapped)
	{
		CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_handle));
		m_isMapped = true;
	}
}

void GraphicsResource::unmap()
{
	if (m_isMapped)
	{
		CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_handle));
		m_isMapped = false;
	}
}

// Getters
cudaGraphicsResource_t GraphicsResource::getHandle() const
{
    return m_handle;
}

bool GraphicsResource::isRegistered() const
{
	return m_handle != 0;
}

bool GraphicsResource::isMapped() const
{
	return m_isMapped;
}

}
}

