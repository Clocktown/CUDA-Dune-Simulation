#include "grid.cuh"
#include "simulation.cuh"
#include "constant.cuh"
#include <sthe/device/vector_extension.hpp>

namespace dunes
{
namespace device
{

__forceinline__ __device__ int getLinearIndex(const int2& t_cell)
{
	return t_cell.x + t_cell.y * t_simulation.gridSize.x;
}

__forceinline__ __device__ int2 getWrappedCell(const int2& t_cell)
{ 
	return int2{ (t_cell.x + t_simulation.gridSize.x) % t_simulation.gridSize.x,
				 (t_cell.y + t_simulation.gridSize.y) % t_simulation.gridSize.y };
}

__forceinline__ __device__ int2 getNearestCell(const float2& t_position)
{
	return make_int2(roundf(t_position));
}

__forceinline__ __device__ bool isOutside(const int2& t_cell)
{
	return t_cell.x < 0 || t_cell.y < 0 || t_cell.x >= t_simulation.gridSize.x || t_cell.y >= t_simulation.gridSize.y;
}

}
}
