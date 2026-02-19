#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>

namespace dunes
{

	__global__ void setupContinuousSaltationKernel(Array2D<half2> t_terrainArray, const Array2D<half2> t_windArray, Array2D<half4> t_resistanceArray, Buffer<half> t_slabBuffer, Buffer<half> t_advectedSlabBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				float2 terrain{ __half22float2(t_terrainArray.read(cell)) };

				const float2 windVelocity {sampleLinearOrNearest<true>(t_windArray,0.5f * (make_float2(cell) + 0.5f))};
				const float windSpeed{ length(windVelocity) };

				const float4 resistance{ half4toFloat4(t_resistanceArray.read(cell)) };
				const float saltationScale{ (1.0f - resistance.x) * (1.0f - fmaxf(resistance.y, 0.f)) * (resistance.w > 0.0f ? 0.5f : 1.0f) };

				//const float scale{ windSpeed * c_parameters.deltaTime };

				const float saltation{ fminf(c_parameters.saltationStrength * saltationScale + (resistance.w < 0.0f ? -resistance.w : 0.0f), terrain.y) };

				terrain.y -= saltation;
				t_terrainArray.write(cell, __float22half2_rn(terrain));

				const int cellIndex{ getCellIndex(cell) };
				const float slab{ saltation };

				t_slabBuffer[cellIndex] += __float2half(slab);
				t_advectedSlabBuffer[cellIndex] = __float2half(0.0f);
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousSaltationKernel(const Array2D<half2> t_windArray, Buffer<half> t_slabBuffer, Buffer<half> t_advectedSlabBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		const float slab{ __half2float(t_slabBuffer[cellIndex]) };

		const float2 windVelocity { sampleLinearOrNearest<true>(t_windArray,0.5f * (make_float2(cell) + 0.5f))};

		const float2 position{ make_float2(cell) };

		if (slab > 0.0f)
		{
			const float2 nextPosition{ position + windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };

			if constexpr (TUseBilinear) {
				const int2 nextCell{ make_int2(floorf(nextPosition)) };

				for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
				{
					const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

					for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
					{
						const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
						const float weight{ u * v };

						if (weight > 0.0f)
						{
							atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(int2{ x,y })), __float2half(weight * slab));
						}
					}
				}
			}
			else {
				const int2 nextCell{ getNearestCell(nextPosition) };
				atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(nextCell)), __float2half(slab));
			}
		}
	}

	template <bool TUseBilinear>
	__global__ void continuousBackwardSaltationKernel(const Array2D<half2> t_windArray, Buffer<half> t_slabBuffer, Buffer<half> t_advectedSlabBuffer)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		float slab{ 0.f };

		const float2 windVelocity { sampleLinearOrNearest<true>(t_windArray,0.5f * (make_float2(cell) + 0.5f))};

		const float2 position{ make_float2(cell) };

		const float2 nextPosition{ position - windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };

		if constexpr (TUseBilinear) {
			const int2 nextCell{ make_int2(floorf(nextPosition)) };

			for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
			{
				const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

				for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
				{
					const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
					const float weight{ u * v };

					if (weight > 0.0f)
					{
						slab += __half2float(t_slabBuffer[getCellIndex(getWrappedCell(int2{ x,y }))]) * weight;
					}
				}
			}
		}
		else {
			const int2 nextCell{ getNearestCell(nextPosition) };
			slab += __half2float(t_slabBuffer[getCellIndex(getWrappedCell(nextCell))]);
		}


		t_advectedSlabBuffer[cellIndex] = __float2half(slab);
	}

	__global__ void finishContinuousSaltationKernel(Array2D<half2> t_terrainArray, const Array2D<half2> t_windArray, const Array2D<half4> t_resistanceArray, Buffer<half> t_slabBuffer, Buffer<half> t_advectedSlabBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				const int cellIndex{ getCellIndex(cell) };

				float2 terrain{ __half22float2(t_terrainArray.read(cell)) };
				const float slab{ __half2float(t_advectedSlabBuffer[cellIndex]) };

				const float windSpeed {length(sampleLinearOrNearest<true>(t_windArray,0.5f * (make_float2(cell) + 0.5f)))};

				const float4 resistance{ half4toFloat4(t_resistanceArray.read(cell)) };
				const float vegetation = fmaxf(resistance.y, 0.f);
				const float object = resistance.y < 0.f ? 0.f : 1.f;
				const float abrasionScale{ object * c_parameters.abrasionStrength * c_parameters.deltaTime * windSpeed * (1.0f - vegetation) * (1.0f - resistance.z) };
				const float vegetationFactor = (terrain.y > 0.0f ? 0.4f : 0.6f);
				const float depositionProbability = object * fminf(fmaxf(fmaxf(resistance.x,
					(1.0f - vegetationFactor) + vegetation * vegetationFactor), resistance.w), resistance.w < 0.f ? 0.f : 1.f);


				const float new_slab = slab * (1.f - depositionProbability);
				float abrasion{ terrain.y < c_parameters.abrasionThreshold && new_slab > 0.f ? abrasionScale * (1.f - depositionProbability) : 0.0f };

				terrain.y += abrasion;
				terrain.x -= abrasion;
				//}
				terrain.y += slab * depositionProbability;
				t_terrainArray.write(cell, __float22half2_rn(terrain));
				t_slabBuffer[cellIndex] = __float2half(slab * (1.f - depositionProbability)); // write updated advectedSlabBuffer back to slabBuffer (ping-pong)
				t_advectedSlabBuffer[cellIndex] = __float2half(slab * (1.f - vegetation)); // Used in Reptation as slabBuffer
			}
		}
	}

	void continuousSaltation(const LaunchParameters& t_launchParameters)
	{
		// TODO: implement Backward saltation (saltationMode)
		setupContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
		if (t_launchParameters.saltationMode == SaltationMode::Backward) {
			if (t_launchParameters.useBilinear) {
			continuousBackwardSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
			}
			else {
				continuousBackwardSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
			}
		}
		else {
			if (t_launchParameters.useBilinear) {
			continuousSaltationKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
			}
			else {
				continuousSaltationKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
			}
		}

		finishContinuousSaltationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
	}

}
