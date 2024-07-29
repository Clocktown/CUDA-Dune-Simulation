#include "constants.cuh"
#include "kernels.cuh"
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

__global__ void setupContinuousReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_reptationBuffer[cellIndex] = 0.0f;
	}
}

__global__ void continuousAngularReptationKernel(const Array2D<float4> t_resistanceArray, const Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer, const Array2D<float2> t_windArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 resistance{ t_resistanceArray.read(cell).x, t_resistanceArray.read(cell).y };
	const float windShadow{ resistance.x * c_parameters.reptationUseWindShadow };

	const float slab{ t_slabBuffer[cellIndex] };
	const float2 wind{ t_windArray.read(cell) };

	float baseAngle = c_parameters.avalancheAngle * exp(-slab * (1.f - windShadow) * length(wind) * c_parameters.reptationStrength);

	// Store precomputed angle
	t_reptationBuffer[cellIndex] = lerp(baseAngle, c_parameters.vegetationAngle, fmaxf(resistance.y, 0.f));
}

__global__ void noReptationKernel(const Array2D<float4> t_resistanceArray, Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float vegetation{ t_resistanceArray.read(cell).y };

	// Store precomputed angle
	t_reptationBuffer[cellIndex] = lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(vegetation, 0.f));
}

__global__ void continuousReptationKernel(const Array2D<float2> t_terrainArray, Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer, const Array2D<float2> t_windArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };

	const float slab{ t_slabBuffer[cellIndex] };
	const float wind{ length(t_windArray.read(cell)) };

	float change{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float nextSlab{ t_slabBuffer[getCellIndex(nextCell)] };
		const float nextWind{ length(t_windArray.read(cell)) };

		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ (nextHeight - height) * c_parameters.rGridScale * c_rDistances[i]};
		const float heightScale = abs(heightDifference);// fmaxf(c_parameters.avalancheAngle - abs(heightDifference), 0.f) / c_parameters.avalancheAngle;

		// Enforce symmetric additive and subtractive changes, avoiding any atomics
		float step = fmaxf(0.25f * heightScale * (slab + nextSlab) * (wind + nextWind) * c_parameters.reptationSmoothingStrength, 0.f);
        change += signbit(heightDifference) ? -fminf(step, terrain.y) : fminf(step, nextTerrain.y);
	}

	t_reptationBuffer[cellIndex] = change * 0.125;
}

__global__ void continuousBilinearReptationKernel(const Array2D<float2> t_terrainArray, const Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer, const Array2D<float2> t_windArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };

	float slab{ t_slabBuffer[cellIndex] };
	float2 wind{ t_windArray.read(cell) };

	float change{ 0.0f };

	constexpr float2 sobel[8]{ {0,-2}, {1, -1}, {2, 0}, {1, 1}, {0, 2}, { -1, 1}, { -2, 0}, { -1, -1} };
	float2 gradient{ 0,0 };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float nextSlab{ t_slabBuffer[getCellIndex(nextCell)] };

		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		// Want the negative gradient
		gradient -= sobel[i] * nextHeight;
	}

	const float gradientStrength = length(gradient);
	float scale = fminf(slab * gradientStrength * length(wind) * c_parameters.reptationStrength, terrain.y);

	const float2 offset{ 1.f * gradient * c_parameters.deltaTime / (gradientStrength + 1e-5f) };

	if (scale > 0.0f)
	{
		const float2 position{ make_float2(cell) + offset};
		const int2 nextCell{ make_int2(floorf(position)) };

		for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
		{
			const float u{ 1.0f - abs(static_cast<float>(x) - position.x) };

			for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
			{
				const float v{ 1.0f - abs(static_cast<float>(y) - position.y) };
				const float weight{ u * v };

				if (weight > 0.0f)
				{
					atomicAdd(t_reptationBuffer + getCellIndex(getWrappedCell(int2{ x,y })), weight * scale);
				}
			}
		}
		atomicAdd(t_reptationBuffer + cellIndex, -scale);
	}
}

__global__ void finishContinuousReptationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_reptationBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			float2 terrain{ t_terrainArray.read(cell) };
			terrain.y += t_reptationBuffer[getCellIndex(cell)];

			t_terrainArray.write(cell, terrain);
		}
	}
}

__global__ void finishContinuousBilinearReptationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_reptationBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			float2 terrain{ t_terrainArray.read(cell) };
			terrain.y += t_reptationBuffer[cellIndex];

			t_terrainArray.write(cell, terrain);
		}
	}
}

void continuousReptation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };
	if (t_simulationParameters.reptationSmoothingStrength > 0.f) {
		continuousReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.tmpBuffer, reptationBuffer, t_launchParameters.windArray);
		finishContinuousReptationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, reptationBuffer);
	}
	if (t_simulationParameters.reptationStrength > 0.f) {
		continuousAngularReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer, reptationBuffer, t_launchParameters.windArray);
	}
	else {
		noReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, reptationBuffer);
	}
}

}
