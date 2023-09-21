#include "multigrid.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>

#define RSQRT2Pi 0.3989422804f

namespace dunes
{

__forceinline__ __device__ float gaussian(const float t_x, const float t_sigma)
{
	const float rsigma{ 1.0f / t_sigma };
	return RSQRT2Pi * rsigma * expf(-0.5f * t_x * t_x * rsigma * rsigma);
}

__global__ void initializeWindWarpingKernel(WindWarping t_windWarping)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	const float2 center{ 0.5f * make_float2(c_parameters.gridSize) };
	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			int2 fftshift{ 0, 0 };

			if (cell.x > center.x)
			{
				fftshift.x = c_parameters.gridSize.x;
			}
			if (cell.y > center.y)
			{
				fftshift.y = c_parameters.gridSize.y;
			}

			const float distance{ length(c_parameters.gridScale * (make_float2(cell - fftshift) + 0.5f)) };
		
			for (int i{ 0 }; i < t_windWarping.count; ++i)
			{
				t_windWarping.gaussKernels[i][cellIndex] = cuComplex{ gaussian(distance, 0.5f * t_windWarping.radii[i]), 0.0f };
			}
		}
	}
}

__global__ void setupWindWarpingKernel(Array2D<float2> t_terrainArray, Buffer<cuComplex> t_heightBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	const float2 center{ 0.5f * make_float2(c_parameters.gridSize) };
	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };
			const float2 terrain{ t_terrainArray.read(cell) };
			const float height{ terrain.x + terrain.y };

			t_heightBuffer[cellIndex] = cuComplex{ height, 0.0f };
		}
	}
}

__global__ void smoothTerrainsKernel(Buffer<cuComplex> t_heightBuffer, WindWarping t_windWarping)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };
			const cuComplex height{ t_heightBuffer[cellIndex] };
			const float fftScale{ 1.0f / static_cast<float>(c_parameters.gridSize.x * c_parameters.gridSize.y) };

			for (int i{ 0 }; i < t_windWarping.count; ++i)
			{
				t_windWarping.smoothedHeights[i][cellIndex] = fftScale * t_windWarping.gaussKernels[i][cellIndex] * height;
			}
		}
	}
}

__global__ void windWarpingKernel(Array2D<float2> t_windArray, WindWarping t_windWarping)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };
			const float2 windVelocity{ t_windArray.read(cell) };
			const float windSpeed{ length(windVelocity) };
			const float2 windDirection{ windVelocity / (windSpeed + 0.000001f) };
			
			float2 warpDirection{ 0.0f, 0.0f };
			float weight{ 0.0f };

			for (int i{ 0 }; i < t_windWarping.count; ++i)
			{
				const float smoothedHeights[4]{ t_windWarping.smoothedHeights[i][getCellIndex(getWrappedCell(cell + int2{ -1, 0 }))].x,
								                t_windWarping.smoothedHeights[i][getCellIndex(getWrappedCell(cell + int2{ 1, 0 }))].x,
								                t_windWarping.smoothedHeights[i][getCellIndex(getWrappedCell(cell + int2{ 0, -1 }))].x,
								                t_windWarping.smoothedHeights[i][getCellIndex(getWrappedCell(cell + int2{ 0, 1 }))].x };

				const float scale{ t_windWarping.i_divisor * 0.5f * c_parameters.rGridScale };
				const float2 gradient{ scale * (smoothedHeights[1] - smoothedHeights[0]),
								       scale * (smoothedHeights[3] - smoothedHeights[2]) };

				const float gradientLength{ length(gradient) };
				
				float2 orthogonalDirection{ -gradient.y, gradient.x };
				orthogonalDirection *= sign(dot(windDirection, orthogonalDirection));
				
				float alpha{ fminf(gradientLength, 1.0f) }; 
			
				warpDirection += t_windWarping.strengths[i] * lerp(windDirection, t_windWarping.gradientStrengths[i] * orthogonalDirection, alpha);
				weight += t_windWarping.strengths[i];
			}

			if (weight > 0.0f)
			{
				warpDirection /= weight;
			}

			warpDirection /= (length(warpDirection) + 0.000001f);
			t_windArray.write(cell, warpDirection * windSpeed);
		}
	}
}

void initializeWindWarping(const LaunchParameters& t_launchParameters)
{
	initializeWindWarpingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.windWarping);

	for (int i{ 0 }; i < t_launchParameters.windWarping.count; ++i)
	{
		CUFFT_CHECK_ERROR(cufftExecC2C(t_launchParameters.fftPlan, t_launchParameters.windWarping.gaussKernels[i], t_launchParameters.windWarping.gaussKernels[i], CUFFT_FORWARD));
	}
}

void windWarping(const LaunchParameters& t_launchParameters)
{
	if (t_launchParameters.windWarpingMode == WindWarpingMode::Standard)
	{
		Buffer<cuComplex> heightBuffer{ reinterpret_cast<Buffer<cuComplex>>(t_launchParameters.tmpBuffer) };
	
	    setupWindWarpingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, heightBuffer);

	    CUFFT_CHECK_ERROR(cufftExecC2C(t_launchParameters.fftPlan, heightBuffer, heightBuffer, CUFFT_FORWARD));

	    smoothTerrainsKernel<<< t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(heightBuffer, t_launchParameters.windWarping);

	    for (int i{ 0 }; i < t_launchParameters.windWarping.count; ++i)
	    {
		    CUFFT_CHECK_ERROR(cufftExecC2C(t_launchParameters.fftPlan, t_launchParameters.windWarping.smoothedHeights[i], t_launchParameters.windWarping.smoothedHeights[i], CUFFT_INVERSE));
	    }

	    windWarpingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.windArray, t_launchParameters.windWarping);
	}
}

}
