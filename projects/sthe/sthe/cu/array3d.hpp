#pragma once

#include "array.hpp"
#include <sthe/gl/image.hpp>
#include <glad/glad.h>
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
	Array3D(gl::Image& t_image, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	Array3D(const GLuint t_image, const GLenum t_target, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	Array3D(const Array3D& t_array3D) noexcept;
	Array3D(Array3D&& t_array3D) noexcept;

	// Destructor
	~Array3D();
	
	// Operators
	Array3D& operator=(const Array3D& t_array3D) noexcept;
	Array3D& operator=(Array3D&& t_array3D) noexcept;

	// Functionality
	void reinitialize(const int t_width, const int t_height, const int t_depth, const cudaChannelFormatDesc& t_format, const unsigned int t_flags = cudaArrayDefault);
	void reinitialize(gl::Image& t_image, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	void reinitialize(const GLuint t_image, const GLenum t_target, const unsigned int t_flags = cudaGraphicsRegisterFlagsNone);
	cudaSurfaceObject_t recreateSurface() override;
	cudaTextureObject_t recreateTexture(const cudaTextureDesc& t_descriptor = {}) override;
	void release() override;
	void map(const int t_layer = 0, const int t_mipLevel = 0) override;
	void unmap() override;

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
	cudaSurfaceObject_t getSurface() const override;
	cudaTextureObject_t getTexture() const override;
private:
	// Attributes
	cudaArray_t m_handle;
	int m_width;
	int m_height;
	int m_depth;
	cudaChannelFormatDesc m_format;
	unsigned int m_flags;

	cudaGraphicsResource_t m_graphicsResource;
	bool m_isMapped;

	cudaSurfaceObject_t m_surface;
	cudaTextureObject_t m_texture;
};

}
}

#include "array3D.inl"
