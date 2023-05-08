#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace sthe
{
namespace cu
{

std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, cudaChannelFormatDesc& t_format);

class Array
{
public:
	// Constructors
	Array() = default;
	Array(const Array& t_array) = delete;
	Array(Array&& t_array) = delete;

	// Destructor
	virtual ~Array() = default;

	// Operators
	Array& operator=(const Array& t_array) = delete;
	Array& operator=(Array&& t_array) = delete;

	// Functionality
	virtual void release() = 0;

	// Getters
	virtual cudaArray_t getHandle() const = 0;
	virtual int getWidth() const;
	virtual int getHeight() const;
	virtual int getDepth() const;
	virtual const cudaChannelFormatDesc& getFormat() const = 0;
	int getStride() const;
	virtual unsigned int getFlags() const = 0;
	bool hasStorage() const;
	virtual bool isMapped() const = 0;
};

}
}
