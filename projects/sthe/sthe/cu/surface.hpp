#pragma once

#include "array.hpp"
#include <cuda_runtime.h>

namespace sthe
{
namespace cu
{

class Surface
{
public:
	// Constructors
	Surface();
	explicit Surface(Array& t_array);
	explicit Surface(const cudaArray_t t_array);
	Surface(const Surface& t_surface) = delete;
	Surface(Surface&& t_surface) noexcept;

	// Destructor
	~Surface();

	// Operators
	Surface& operator=(const Surface& t_surface) = delete;
	Surface& operator=(Surface&& t_surface) noexcept;

	// Functionality
	void reinitialize(Array& t_array);
	void reinitialize(const cudaArray_t t_array);
	void release();

	// Getters
	cudaSurfaceObject_t getHandle() const;
	bool hasResource() const;
private:
	// Attribute
	cudaSurfaceObject_t m_handle;
};

}
}
