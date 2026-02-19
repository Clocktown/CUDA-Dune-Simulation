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

	const int2 center{ c_parameters.windGridSize / 2 };
	int2 cell;
    const int  width {2 * (c_parameters.windGridSize.x / 2 + 1)};

	for (cell.x = index.x; cell.x < c_parameters.windGridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.windGridSize.y; cell.y += stride.y)
		{
            const int cellIndex {cell.x + cell.y * width};

			int2 fftshift{ 0, 0 };

			if (cell.x >= center.x)
			{
				fftshift.x = c_parameters.windGridSize.x;
			}
			if (cell.y >= center.y)
			{
				fftshift.y = c_parameters.windGridSize.y;
			}

			const float distance{ length(c_parameters.windGridScale * (make_float2(cell  - fftshift) + 0.5f))};
		
			for (int i{ 0 }; i < t_windWarping.count; ++i)
			{
				((float*)t_windWarping.gaussKernels[i])[cellIndex] = gaussian(distance, 0.5f * t_windWarping.radii[i]);
			}
		}
	}
}

__global__ void setupWindWarpingKernel(Array2D<half2> t_terrainArray, Buffer<float> t_heightBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	const float2 center{ 0.5f * make_float2(c_parameters.windGridSize) };
	int2 cell;
    const int    width {2 * (c_parameters.windGridSize.x / 2 + 1)};

	for (cell.x = index.x; cell.x < c_parameters.windGridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.windGridSize.y; cell.y += stride.y)
		{
            const int    cellIndex {cell.x + cell.y * width};
            const float2 terrain {sampleLinearOrNearest<true>(
                    t_terrainArray, make_float2(2 * cell + 1) + 0.5f)};
			const float height{ terrain.x + terrain.y };

			t_heightBuffer[cellIndex] = height;
		}
	}
}

__global__ void smoothTerrainsKernel(Buffer<cuComplex> t_heightBuffer, WindWarping t_windWarping)
{
    const int2 cell {getGlobalIndex2D()};
    const int2 size {c_parameters.windGridSize.x / 2 + 1, c_parameters.windGridSize.y};

	if(isOutside(cell, size))
    {
        return;
    }

    const int       cellIndex {getCellIndex(cell, size)};
	const cuComplex height{ t_heightBuffer[cellIndex] };
	const float fftScale{ 1.0f / static_cast<float>(c_parameters.windGridSize.x * c_parameters.windGridSize.y) };

	for (int i{ 0 }; i < t_windWarping.count; ++i)
	{
		const cuComplex gauss = t_windWarping.gaussKernels[i][cellIndex];
		const cuComplex result = { // Complex Multiplication
			gauss.x * height.x - gauss.y * height.y, 
			gauss.x * height.y + gauss.y * height.x 
		};
		t_windWarping.smoothedHeights[i][cellIndex] = fftScale * result;
	}
}

__global__ void scaleGaussKernel(float* t_gauss, float scale) {
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;
    const int width {2 * (c_parameters.windGridSize.x / 2 + 1)};

	for (cell.x = index.x; cell.x < c_parameters.windGridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.windGridSize.y; cell.y += stride.y)
		{
            const int cellIndex {cell.x + cell.y * width};
			t_gauss[cellIndex] *= scale;
		}
	}
}

__global__ void windWarpingKernel(Array2D<half2> t_windArray, WindWarping t_windWarping)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };
	int2 cell;
    const int width {2 * (c_parameters.windGridSize.x / 2 + 1)};
    const int2 size {width, c_parameters.windGridSize.y};

	for (cell.x = index.x; cell.x < c_parameters.windGridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.windGridSize.y; cell.y += stride.y)
		{
            //const int    cellIndex {cell.x + cell.y * width};
            const float2 windVelocity{ __half22float2(t_windArray.read(cell)) };
			const float windSpeed{ length(windVelocity) };
			//const float2 windDirection{ windVelocity / (windSpeed + 0.000001f) };
			
			float2 warpDirection{ 0.0f, 0.0f };
			float weight{ 0.0f };

			for (int i{ 0 }; i < t_windWarping.count; ++i)
			{
                const float smoothedHeights[4] {
                        ((float*)t_windWarping
                                .smoothedHeights[i])[getCellIndex(
                                        getWrappedCell(cell + int2 {-1, 0},
                                                        c_parameters.windGridSize),
                                        size)]
                                ,
                        ((float*)t_windWarping
                                .smoothedHeights[i])[getCellIndex(
                                        getWrappedCell(cell + int2 {1, 0},
                                                        c_parameters.windGridSize),
                                        size)]
                                ,
                        ((float*)t_windWarping
                                .smoothedHeights[i])[getCellIndex(
                                        getWrappedCell(cell + int2 {0, -1},
                                                        c_parameters.windGridSize),
                                        size)]
                                ,
                        ((float*)t_windWarping
                                .smoothedHeights[i])[getCellIndex(
                                        getWrappedCell(cell + int2 {0, 1},
                                                        c_parameters.windGridSize),
                                        size)]
                                };

				const float scale{ t_windWarping.i_divisor * 0.5f * c_parameters.rWindGridScale };
				const float2 gradient{ scale * (smoothedHeights[1] - smoothedHeights[0]),
								       scale * (smoothedHeights[3] - smoothedHeights[2]) };

				const float gradientLength{ length(gradient) };
				
				float2 orthogonalDirection{ -gradient.y, gradient.x };
				orthogonalDirection *= sign(dot(windVelocity, orthogonalDirection));
				
				float alpha{ fminf(gradientLength, 1.0f) }; 
			
				warpDirection += t_windWarping.strengths[i] * lerp(windVelocity, t_windWarping.gradientStrengths[i] * orthogonalDirection, alpha);
				weight += t_windWarping.strengths[i];
			}

			if (weight > 0.0f)
			{
				warpDirection /= weight;
			}

			warpDirection /= (length(warpDirection) + 0.000001f);
			t_windArray.write(cell, __float22half2_rn(warpDirection * windSpeed));
		}
	}
}

void initializeWindWarping(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
    int fixedCellCount = t_simulationParameters.windGridSize.y * 2 *
                         (t_simulationParameters.windGridSize.x / 2 + 1);
    for(int i = 0; i < t_launchParameters.windWarping.count; ++i)
    {
        cudaMemset(
                t_launchParameters.windWarping.gaussKernels[i], 0, fixedCellCount * sizeof(float));
    }
	initializeWindWarpingKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.windWarping);

	// Normalize Kernels
	for (int i = 0; i < t_launchParameters.windWarping.count; ++i) {
		float result = thrust::reduce(thrust::device, (float*)t_launchParameters.windWarping.gaussKernels[i], ((float*)t_launchParameters.windWarping.gaussKernels[i]) + fixedCellCount);
            scaleGaussKernel<<<t_launchParameters.optimalGridSize2D,
                               t_launchParameters.optimalBlockSize2D>>>(
                    (float*)t_launchParameters.windWarping.gaussKernels[i], 1.f / result);
	}

	for (int i{ 0 }; i < t_launchParameters.windWarping.count; ++i)
	{
		CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.fftPlanR2C, (cufftReal*)t_launchParameters.windWarping.gaussKernels[i], t_launchParameters.windWarping.gaussKernels[i]));
	}
}

void windWarping(const LaunchParameters&     t_launchParameters,
                 const SimulationParameters& t_simulationParameters)
{
	if (t_launchParameters.windWarpingMode == WindWarpingMode::Standard)
	{
		Buffer<cuComplex> heightBuffer{ reinterpret_cast<Buffer<cuComplex>>(t_launchParameters.tmpBuffer) };
        Buffer<float> heightBufferReal {
                    reinterpret_cast<Buffer<float>>(t_launchParameters.tmpBuffer)};
	    setupWindWarpingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, heightBufferReal);

	    CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.fftPlanR2C, (cufftReal*)heightBuffer, heightBuffer));

		dim3 gridSize;
        gridSize.x = static_cast<unsigned int>(ceilf(
                static_cast<float>(t_simulationParameters.windGridSize.x / 2 + 1) / 8.0f));
        gridSize.y = static_cast<unsigned int>(
                ceilf(static_cast<float>(t_simulationParameters.windGridSize.y) / 8.0f));
        gridSize.z = 1;

	    smoothTerrainsKernel<<<gridSize, dim3 {8, 8, 1}>>>(heightBuffer,
                                                           t_launchParameters.windWarping);

	    for (int i{ 0 }; i < t_launchParameters.windWarping.count; ++i)
	    {
			// TODO: try C2R to heightBuffer instead? would reduce memory requirements?
			// TODO: Maybe also use R2C? more memory but better performance?
			// TODO: try out half precision? May only work for powers of two
			// TODO: DOWNSAMPLE terrain to half resolution
		    CUFFT_CHECK_ERROR(cufftExecC2R(t_launchParameters.fftPlanC2R, t_launchParameters.windWarping.smoothedHeights[i], (cufftReal*)t_launchParameters.windWarping.smoothedHeights[i]));
	    }

		//readBackSmoothTerrainKernel << < t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windWarping);

	    windWarpingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.windArray, t_launchParameters.windWarping);
	}
}

}
