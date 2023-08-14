#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace dunes
{

__global__ void coverageKernel(Array2D<float2> t_terrainArray, unsigned int* coverageMap, float threshold)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 terrain{ t_terrainArray.read(cell) };

	coverageMap[getCellIndex(cell)] = terrain.y > threshold ? 1 : 0;
}

float coverage(const LaunchParameters& t_launchParameters, unsigned int* coverageMap, int num_cells, float threshold)
{
	coverageKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, coverageMap, threshold);
	unsigned int result = thrust::reduce(thrust::device, coverageMap, coverageMap + num_cells);
	return float(result) / float(num_cells);
}

}
