#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void reptationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_slabBuffer)
{
	const int2 cell{ getGlobalIndex2D() };
	
	if (isOutside(cell))
	{
		return;
	}

	float2 terrain{ t_terrainArray.read(cell) };
	terrain.y += t_slabBuffer[getCellIndex(cell)];

	t_terrainArray.write(cell, terrain);
}

void reptation(const LaunchParameters& t_launchParameters)
{
	reptationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);
}

}
