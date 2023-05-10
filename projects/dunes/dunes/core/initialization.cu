#include "simulation.cuh"
#include "constant.cuh"
#include "grid.cuh"
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void initializeKernel(const cudaSurfaceObject_t t_heightMap)
{
	const int2 cell{ getCell() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 height{ 0.0f, 50.0f * (sinf(10.0f * static_cast<float>(cell.x) / static_cast<float>(c_simulationParameter.gridSize.x)) + 0.5f) };

	surf2Dwrite<float2>(height, t_heightMap, cell.x * static_cast<int>(sizeof(float2)), cell.y);
}

void initialize(const LaunchParameter& t_launchParameter, const cudaSurfaceObject_t t_heightMap)
{
	initializeKernel<<<t_launchParameter.gridSize2D, t_launchParameter.blockSize2D>>>(t_heightMap);
}

}
