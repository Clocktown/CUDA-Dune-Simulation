#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "multigrid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAtomicAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float> t_avalancheBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_avalancheBuffer[cellIndex] = 0.0f;
	}
}

template <bool TUseAvalancheStrength>
__global__ void atomicAvalanchingKernel(Array2D<float2> t_terrainArray, const Array2D<float4> t_resistanceArray, Buffer<float> t_avalancheBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };
	const float avalancheAngle{ lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f)) };

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		nextCellIndices[i] = getCellIndex(nextCell);

		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf((TUseAvalancheStrength ? c_parameters.avalancheStrength : 1.0f) * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };

		atomicAdd(t_avalancheBuffer + cellIndex, -avalancheSize);

		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(t_avalancheBuffer + nextCellIndices[i], scale * avalanches[i]);
			}
		}
	}
}

__global__ void finishAtomicAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float> t_avalancheBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += static_cast<int>(blockDim.x * gridDim.x))
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += static_cast<int>(blockDim.y * gridDim.y))
		{
			const int cellIndex{ getCellIndex(cell) };

			float2 terrain{ t_terrainArray.read(cell) };
			terrain.y += t_avalancheBuffer[cellIndex];

			t_avalancheBuffer[cellIndex] = 0.0f;
			t_terrainArray.write(cell, terrain);
		}
	}
}

__global__ void setupAtomicInPlaceAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainBuffer[cellIndex] = t_terrainArray.read(cell);
}

__global__ void setupJacobiAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float> t_sandBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	float2 terrain = t_terrainArray.read(cell);

	t_sandBuffer[cellIndex] = terrain.y;
}

__global__ void jacobiAvalanchingKernel(const Array2D<float4> t_resistanceArray, const Array2D<float2> t_terrainArray, const Buffer<float> t_reptationBuffer, const Buffer<float> t_oldSandBuffer, Buffer<float> t_newSandBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainArray.read(cell) };
	const float oldSandHeight{ t_oldSandBuffer[cellIndex] };
	const float b{ terrain.x + terrain.y };
	const float bSand{ terrain.y };
	const float height{ terrain.x + oldSandHeight };
	float baseAngle = c_parameters.avalancheAngle;
	if (c_parameters.reptationStrength > 0.f) {
		baseAngle = lerp(0.f, baseAngle, t_reptationBuffer[cellIndex]);
	}
	const float avalancheAngle{ lerp(baseAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f)) };

	float val = 0.f;

	for (int i{ 0 }; i < 8; i+=2)
	{
		const int2 nextCell = getWrappedCell(cell + c_offsets[i]);
		const int nextCellIndex = getCellIndex(nextCell);
		const float2 nextTerrain{ t_terrainArray.read(nextCell)};
		const float nextOldSandHeight{ t_oldSandBuffer[nextCellIndex] };
		const float nextHeight{ nextTerrain.x + nextOldSandHeight };

		float nextBaseAngle = c_parameters.avalancheAngle;
		if (c_parameters.reptationStrength > 0.f) {
			nextBaseAngle = lerp(0.f, baseAngle, t_reptationBuffer[nextCellIndex]);
		}
		const float nextAvalancheAngle{ lerp(nextBaseAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(nextCell).y, 0.f)) };

		float h1{ nextHeight + avalancheAngle * c_parameters.gridScale + 2*b };
		if (height - (nextHeight + avalancheAngle * c_parameters.gridScale) <= 0) {
			h1 = 3*height;
		}
		float h2{ nextHeight - nextAvalancheAngle * c_parameters.gridScale + 2*b };
		if (nextHeight - (height + nextAvalancheAngle * c_parameters.gridScale) <= 0) {
			h2 = 3*height;
		}
		val += h1;
		val += h2;
	}
	val /= 24.f;
	float diff = height - val;
	if (diff > oldSandHeight) {
		//diff = oldSandHeight;
	}

	t_newSandBuffer[cellIndex] = oldSandHeight - diff;
}

template <bool TUseAvalancheStrength>
__global__ void atomicInPlaceAvalanchingKernel(const Array2D<float4> t_resistanceArray, Buffer<float2> t_terrainBuffer, const Buffer<float> t_reptationBuffer)
{
	/*const int2 index{ getGlobalIndex2D() };
	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += static_cast<int>(blockDim.x * gridDim.x))
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += static_cast<int>(blockDim.y * gridDim.y))
		{
			const int cellIndex{ getCellIndex(cell) };

			const float2 terrain{ t_terrainBuffer[cellIndex] };
			const float height{ terrain.x + terrain.y };
			const float avalancheAngle{ lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f)) };

			int nextCellIndices[8];
			float avalanches[8];
			float avalancheSum{ 0.0f };
			float maxAvalanche{ 0.0f };

			for (int i{ 0 }; i < 8; ++i)
			{
				nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
				const float2 nextTerrain{ t_terrainBuffer[nextCellIndices[i]] };
				const float nextHeight{ nextTerrain.x + nextTerrain.y };

				const float heightDifference{ height - nextHeight };
				avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
				avalancheSum += avalanches[i];
				maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
			}

			if (avalancheSum > 0.0f)
			{
				const float rAvalancheSum{ 1.0f / avalancheSum };
				const float avalancheSize{ fminf((TUseAvalancheStrength ? c_parameters.avalancheStrength : 1.0f) * maxAvalanche /
												 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };


				const float scale{ avalancheSize * rAvalancheSum };

				for (int i{ 0 }; i < 8; ++i)
				{
					if (avalanches[i] > 0.0f)
					{
						atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, scale * avalanches[i]);
					}
				}

				atomicAdd(&t_terrainBuffer[cellIndex].y, -avalancheSize);
			}
		}
	}*/

	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainBuffer[cellIndex] };
	const float height{ terrain.x + terrain.y };
	float baseAngle = c_parameters.avalancheAngle;
	if (c_parameters.reptationStrength > 0.f) {
		baseAngle = lerp(0.f, baseAngle, t_reptationBuffer[cellIndex]);
	}
	const float avalancheAngle{ lerp(baseAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f)) };

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
		const float2 nextTerrain{ t_terrainBuffer[nextCellIndices[i]] };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf((TUseAvalancheStrength ? c_parameters.avalancheStrength : 1.0f) * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, scale * avalanches[i]);
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].y, -avalancheSize);
	}
}

__global__ void atomicInPlaceTaylorKernel(const Array2D<float4> t_resistanceArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainBuffer[cellIndex] };
	const float height{ terrain.x + terrain.y };
	const float avalancheAngle{ lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f)) };

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxTan{ 0.0f };
	float Bmax{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
		const float2 nextTerrain{ t_terrainBuffer[nextCellIndices[i]] };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		const float tanToNeighbor{ heightDifference * c_rDistances[i] * c_parameters.rGridScale };
		if (tanToNeighbor > avalancheAngle) {
			avalanches[i] = tanToNeighbor;
			avalancheSum += avalanches[i];
		}
		else {
			avalanches[i] = 0.f;
		}
		
		if (avalanches[i] > maxTan) {
			maxTan = avalanches[i];
			Bmax = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.f);
		}
	}

	if (avalancheSum > 0.0f)
	{
		const float k_c = c_parameters.avalancheStrength;
		const float rAvalancheSum{ 1.0f / avalancheSum };


		const float scale{ k_c * Bmax * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, scale * avalanches[i]);
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].y, -k_c * Bmax);
	}
}

__device__ __inline__ int linear_block_10x10(int x, int y) { return (y + 1) * 10 + (x + 1); }

template <bool TUseAvalancheStrength>
__global__ void sharedAtomicInPlaceAvalanchingKernel(const Array2D<float4> t_resistanceArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };
	const int2 baseID = int2{ int(blockIdx.x * blockDim.x), int(blockIdx.y * blockDim.y) };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	__shared__ float s[10 * 10];
	const int2 threadID = int2{ int(threadIdx.x), int(threadIdx.y) };
	// threadIndx is the index in the innermost 6x6 area calculated as if it was a 6x6 work group.
	// These 36 innermost threads need no atomic add since they exclude the border.
	// Coincidentally, the additional outer border that fills from 8x8 to 10x10 is exactly 36 cells.
	// These need additional atomicWrites, which is why the inner 6x6 threads load and write this border region.
	const int              threadIndx = ((threadIdx.x % 7) == 0 || (threadIdx.y % 7) == 0) ? 36 : (threadIdx.y - 1) * 6 + (threadIdx.x - 1);
	const int              idx = cellIndex;
	const int              idx_shared = linear_block_10x10(threadIdx.x, threadIdx.y);
	const float2           terrain = t_terrainBuffer[idx];
	const float avalancheAngle = lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.f));

	// Load into Shared memory
	s[idx_shared] = terrain.x + terrain.y;
	int2 off{ 0, 0 }; // I wish I could make this const...
	if (threadIndx < 10)
	{ // leftmost column
		off = int2{ -1, threadIndx - 1 };
	}
	else if (threadIndx < 20)
	{ // rightmost column
		off = int2{ 8, threadIndx - 11 };
	}
	else if (threadIndx < 28)
	{ // top row (minus edge columns)
		off = int2{ threadIndx - 20, 8 };
	}
	else if (threadIndx < 36)
	{ // bottom row (minus edge columns)
		off = int2{ threadIndx - 28, -1 };
	}

	const int2 b = getWrappedCell(baseID + off);
	const int idxAtomic = (threadIndx >= 36) ? idx : getCellIndex(b);
	const int idxSharedAtomic = (threadIndx >= 36) ? idx_shared : linear_block_10x10(off.x, off.y);

	if (threadIndx < 36)
	{
		float2 v = t_terrainBuffer[idxAtomic];
		s[idxSharedAtomic] = v.x + v.y;
	}
	__syncthreads();

	const float height{ terrain.x + terrain.y };

	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		int2 b = threadID + c_offsets[i];
		//nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
		const float nextHeight{ s[linear_block_10x10(b.x, b.y)] };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	__syncthreads();
	s[idx_shared] = 0.f;
	if (threadIndx < 36)
	{
		s[idxSharedAtomic] = 0.f;
	}
	__syncthreads();

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf((TUseAvalancheStrength ? c_parameters.avalancheStrength : 1.0f) * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i = 0; i < 8; ++i)
		{
			if (avalanches[i] <= 0)
				continue;
			avalanches[i] *= scale;
			const int2 b = threadID + c_offsets[i];
			atomicAdd(&s[linear_block_10x10(b.x, b.y)], avalanches[i]);
		}
		atomicAdd(&s[idx_shared], -avalancheSize);
	}

	__syncthreads();
	atomicAdd(&t_terrainBuffer[idxAtomic].y, s[idxSharedAtomic]);

	if (threadIndx < 36)
	{
		// 6x6 inner region
		t_terrainBuffer[idx].y = terrain.y + s[idx_shared];
	}
}

__global__ void finishJacobiAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float> t_sandBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 terrain{ t_terrainArray.read(cell) };

	t_terrainArray.write(cell, float2{ terrain.x ,t_sandBuffer[cellIndex] });
}

__global__ void finishAtomicInPlaceAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainArray.write(cell, t_terrainBuffer[cellIndex]);
}

void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	Buffer<float2> terrainBuffer{ reinterpret_cast<Buffer<float2>>(t_launchParameters.tmpBuffer) };
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };
	Buffer<float> oldSandBuffer{ t_launchParameters.tmpBuffer };
	Buffer<float> newSandBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };

	// Fill bBuffer with height, oldSandBuffer with sand
	setupJacobiAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, oldSandBuffer);

	// "Jacobi"-Iterations
	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i) {
		jacobiAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, t_launchParameters.terrainArray, reptationBuffer, oldSandBuffer, newSandBuffer);
		std::swap(oldSandBuffer, newSandBuffer);
	}

	// copy sandBuffer to actual surface
	finishJacobiAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, oldSandBuffer);

	return;

	switch (t_launchParameters.avalancheMode)
	{
	case AvalancheMode::AtomicBuffered:
		setupAtomicAvalanchingKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);

	    for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	    {
			if (i % t_launchParameters.avalancheSoftIterationModulus == 0 || 
				i >= t_launchParameters.avalancheIterations - t_launchParameters.avalancheFinalSoftIterations) 
			{
				atomicAvalanchingKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
			}
			else {
				atomicAvalanchingKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
			}
			finishAtomicAvalanchingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);
	    } 

	    break;
	case AvalancheMode::AtomicInPlace:
		setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
		{
			if (i % t_launchParameters.avalancheSoftIterationModulus == 0 ||
				i >= t_launchParameters.avalancheIterations - t_launchParameters.avalancheFinalSoftIterations) 
			{
				atomicInPlaceAvalanchingKernel<true><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, terrainBuffer, reptationBuffer);
			}
			else
			{
				atomicInPlaceAvalanchingKernel<false><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, terrainBuffer, reptationBuffer);
			}
		}

		finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		break;
	case AvalancheMode::SharedAtomicInPlace:
		setupAtomicInPlaceAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, terrainBuffer);

		for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
		{
			if (i % t_launchParameters.avalancheSoftIterationModulus == 0 ||
				i >= t_launchParameters.avalancheIterations - t_launchParameters.avalancheFinalSoftIterations)
			{
				sharedAtomicInPlaceAvalanchingKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, terrainBuffer);
			}
			else
			{
				sharedAtomicInPlaceAvalanchingKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, terrainBuffer);
			}
		}

		finishAtomicInPlaceAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, terrainBuffer);

		break;
	case AvalancheMode::MixedInPlace:
		setupAtomicInPlaceAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, terrainBuffer);

		for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
		{
			if (i % t_launchParameters.avalancheSoftIterationModulus == 0 ||
				i >= t_launchParameters.avalancheIterations - t_launchParameters.avalancheFinalSoftIterations)
			{
				sharedAtomicInPlaceAvalanchingKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, terrainBuffer);
			}
			else
			{
				atomicInPlaceAvalanchingKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, terrainBuffer, reptationBuffer);
			}
		}

		finishAtomicInPlaceAvalanchingKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, terrainBuffer);

		break;
	case AvalancheMode::Multigrid:
	    multigrid(t_launchParameters);

	    break;
	case AvalancheMode::Taylor:
		setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);
		for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
		{
			atomicInPlaceTaylorKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, terrainBuffer);
		}

		finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

	}
}

}
