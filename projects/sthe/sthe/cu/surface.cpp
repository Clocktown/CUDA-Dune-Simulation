#include "surface.hpp"
#include "array.hpp"
#include <sthe/config/debug.hpp>
#include <cuda_runtime.h>
#include <utility>

namespace sthe
{
namespace cu
{

// Constructor
Surface::Surface() :
	m_handle{ 0 }
{

}

Surface::Surface(Array& t_array) :
	Surface{ t_array.getHandle() }
{

}

Surface::Surface(const cudaArray_t t_array)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
									 .res{ .array{ .array{ t_array } } } };

	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_handle, &resource));
}

Surface::Surface(Surface&& t_surface) noexcept :
	m_handle{ std::exchange(t_surface.m_handle, 0) }
{

}

// Destructor
Surface::~Surface()
{
	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
}

// Operator
Surface& Surface::operator=(Surface&& t_surface) noexcept
{
	if (this != &t_surface)
	{
		CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));

		m_handle = std::exchange(t_surface.m_handle, 0);
	}

	return *this;
}

// Functionality
void Surface::reinitialize(Array& t_array)
{
	reinitialize(t_array.getHandle());
}

void Surface::reinitialize(const cudaArray_t t_array)
{
	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));

	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
									 .res{ .array{ .array{ t_array } } } };

	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_handle, &resource));
}

void Surface::release()
{
	CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
}

// Getters
cudaSurfaceObject_t Surface::getHandle() const
{
	return m_handle;
}

bool Surface::hasResource() const
{
	return m_handle == 0;
}

}
}
