#pragma once

#include "buffer.hpp"
#include "array2d.hpp"
#include "array3d.hpp"
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/image.hpp>
#include <glad/glad.h>
#include <cuda_runtime.h>

namespace sthe
{
namespace cu
{

class GraphicsResource
{
public:
	// Constructors
	GraphicsResource();
	GraphicsResource(const GraphicsResource& t_graphicsResource) = delete;
	GraphicsResource(GraphicsResource&& t_graphicsResource) noexcept;

	// Destructor
	~GraphicsResource();
	
	// Operators
	GraphicsResource& operator=(const GraphicsResource& t_graphicsResource) = delete;
	GraphicsResource& operator=(GraphicsResource&& t_graphicsResource) noexcept;

	// Functionality
	void registerBuffer(gl::Buffer& t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void registerBuffer(const GLuint t_buffer, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void registerImage(gl::Image& t_image, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void registerImage(const GLuint t_image, const GLenum t_target, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void unregister();
	void mapBuffer(Buffer& t_buffer);
	void mapBuffer(Buffer& t_buffer, const int t_stride);
	void mapImage(Array2D& t_array2D, const int t_layer = 0, const int t_mipLevel = 0);
	void mapImage(Array3D& t_array3D, const int t_layer = 0, const int t_mipLevel = 0);
	void mapImage(cudaArray_t& t_array, const int t_layer = 0, const int t_mipLevel = 0);
	void map();
	void unmap();

	template<typename T>
	void mapBuffer(T*& t_pointer);

	template<typename T>
	void mapBuffer(T*& t_pointer, int& t_count);

	// Getters
	cudaGraphicsResource_t getHandle() const;
	bool isRegistered() const;
	bool isMapped() const;
private:
	// Attributes
	cudaGraphicsResource_t m_handle;
	bool m_isMapped;
};

}
}
