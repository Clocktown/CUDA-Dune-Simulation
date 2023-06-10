#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void initializationKernel(Array2D<float2> t_terrainArray, Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}
	 
	const int2 center{ c_parameters.gridSize / 2 };
	const int2 size{ c_parameters.gridSize / 20 };
	const float maxSandHeight{ static_cast<float>(c_parameters.gridSize.x + c_parameters.gridSize.y) * c_parameters.gridScale / 20.0f };

	const float bedrockHeight{ 0.0f };
	const float sandHeight{ abs(center.x - cell.x) < size.x && abs(center.y - cell.y) < size.y ? maxSandHeight : 0.0f };
	
	const float2 terrain{ bedrockHeight, sandHeight };
	t_terrainArray.write(cell, terrain);
	
	const float4 resistance{ 0.0f, 0.0f, 1.0f, 0.0f };
	t_resistanceArray.write(cell, resistance);

	t_slabBuffer[getCellIndex(cell)] = 0.0f;
}

void initialization(const LaunchParameters& t_launchParameters)
{
	initializationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer);
}

}
