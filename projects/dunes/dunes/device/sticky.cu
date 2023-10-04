#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

	template<bool TUseBilinear>
	__global__ void setupStickyKernel(const Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Buffer<float> t_cliffBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				const float2 terrain{ t_terrainArray.read(cell) };
				const float height{ terrain.x + terrain.y };

				const float2 windVelocity = t_windArray.read(cell);
				const float windSpeed = length(windVelocity);
				const float2 windDirection = windVelocity / (windSpeed + 1e-06f);

				float2 nextPosition{ make_float2(cell) - windDirection + 0.5f };
				const float2 nextTerrain{ sampleLinearOrNearest<TUseBilinear>(t_terrainArray, nextPosition) };
				const float nextHeight{ nextTerrain.x + nextTerrain.y };
				float cliffHeight{ height - nextHeight };
				const float angle{ cliffHeight * c_parameters.rGridScale };

				const int cellIndex{ getCellIndex(cell) };

				if (angle >= c_parameters.stickyAngle)
				{
					t_cliffBuffer[cellIndex] = cliffHeight;
				}
				else
				{
					t_cliffBuffer[cellIndex] = 0.0f;
				}
			}
		}
	}

	template<WindShadowMode Mode, bool TUseBilinear>
	__global__ void stickyKernel(const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> t_cliffBuffer)
	{
		const int2 index{ getGlobalIndex2D() };
		const int2 stride{ getGridStride2D() };

		int2 cell;

		for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
		{
			for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
			{
				float2 nextPosition{ make_float2(cell + 0.5f) };
				const float erosionResistance{ -c_parameters.stickyStrength };
				float4 resistance{ t_resistanceArray.read(cell) };
				resistance.w = 0.f;

				float2 windVelocity;
				float windSpeed;
				float2 windDirection;

				if constexpr (Mode == WindShadowMode::Linear)
				{
					windVelocity = t_windArray.read(cell);
					windSpeed = length(windVelocity);
					windDirection = windVelocity / (windSpeed + 1e-06f);
				}

				for (float distance = c_parameters.gridScale; distance <= c_parameters.stickyRange.y * c_parameters.maxStickyHeight; distance += c_parameters.gridScale)
				{
					if constexpr (Mode == WindShadowMode::Curved)
					{
						windVelocity = sampleLinearOrNearest<TUseBilinear>(t_windArray, nextPosition);;
						windSpeed = length(windVelocity);
						windDirection = windVelocity / (windSpeed + 1e-06f);
					}

					nextPosition += windDirection;

					const int2 nextCell{ getNearestCell(nextPosition - 0.5f) };
					const int nextCellIndex{ getCellIndex(getWrappedCell(nextCell)) };
					const float cliffHeight{ t_cliffBuffer[nextCellIndex] };
					const float correctedDistance = c_parameters.gridScale * length(make_float2(cell - nextCell));

					if (cliffHeight > 0.0f)
					{
						const float maxDistance{ fminf(cliffHeight, c_parameters.maxStickyHeight) };
						const float erosionDistance{ c_parameters.stickyRange.x * maxDistance };
						const float stickyDistance{ c_parameters.stickyRange.y * maxDistance };

						if (correctedDistance <= erosionDistance)
						{
							resistance.w = erosionResistance;
							break;
						}
						else if (correctedDistance <= stickyDistance)
						{
							resistance.w = fmaxf(fminf(0.1 + 1.0f - (correctedDistance - erosionDistance) / (stickyDistance - erosionDistance), 1.f), resistance.w);
						}
					}
				}
				t_resistanceArray.write(cell, resistance);
			}
		}
		/*const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float2 nextPosition{ make_float2(cell + 0.5f) };
		const float erosionResistance{ -c_parameters.stickyStrength };
		float4 resistance{ t_resistanceArray.read(cell) };
		resistance.w = 0.f;

		float2 windVelocity;
		float windSpeed;
		float2 windDirection;

		if constexpr (Mode == WindShadowMode::Linear) {
			windVelocity = t_windArray.read(cell);
			windSpeed = length(windVelocity);
			windDirection = windVelocity / (windSpeed + 1e-06f);
		}

		for (float distance = c_parameters.gridScale; distance <= c_parameters.stickyRange.y * c_parameters.maxStickyHeight; distance += c_parameters.gridScale)
		{
			if constexpr (Mode == WindShadowMode::Curved) {
				windVelocity = sampleLinearOrNearest<TUseBilinear>(t_windArray, nextPosition);;
				windSpeed = length(windVelocity);
				windDirection = windVelocity / (windSpeed + 1e-06f);
			}

			nextPosition += windDirection;

			const int2 nextCell{ getNearestCell(nextPosition - 0.5f) };
			const int nextCellIndex{ getCellIndex(getWrappedCell(nextCell)) };
			const float cliffHeight{ t_cliffBuffer[nextCellIndex] };
			const float correctedDistance = c_parameters.gridScale * length(make_float2(cell - nextCell));

			if (cliffHeight > 0.0f)
			{
				const float maxDistance{ fminf(cliffHeight, c_parameters.maxStickyHeight) };
				const float erosionDistance{ c_parameters.stickyRange.x * maxDistance };
				const float stickyDistance{ c_parameters.stickyRange.y * maxDistance };

				if (correctedDistance <= erosionDistance)
				{
					resistance.w = erosionResistance;
					break;
				}
				else if (correctedDistance <= stickyDistance)
				{
					resistance.w = fmaxf(fminf(0.1 + 1.0f - (correctedDistance - erosionDistance) / (stickyDistance - erosionDistance), 1.f), resistance.w);
					//break;
				}
			}
		}
		t_resistanceArray.write(cell, resistance);*/
	}

	void sticky(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
	{
		if (t_simulationParameters.stickyStrength > 0.0f)
		{
			if (t_launchParameters.useBilinear) {
				setupStickyKernel<true> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.tmpBuffer);
			}
			else {
				setupStickyKernel<false> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.tmpBuffer);
			}
			if (t_launchParameters.windShadowMode == WindShadowMode::Linear)
			{
				if (t_launchParameters.useBilinear)
					stickyKernel<WindShadowMode::Linear, true> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
				else
					stickyKernel<WindShadowMode::Linear, false> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
			}
			else
			{
				if (t_launchParameters.useBilinear)
					stickyKernel<WindShadowMode::Curved, true> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
				else
					stickyKernel<WindShadowMode::Curved, false> << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
			}
		}
	}

}
