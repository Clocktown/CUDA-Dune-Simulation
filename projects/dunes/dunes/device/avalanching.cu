#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
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

__global__ void atomicAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float> t_avalancheBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };

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
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf(c_parameters.avalancheStrength * maxAvalanche /
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

template <bool TUseAvalancheStrength>
__global__ void atomicInPlaceAvalanchingKernel(Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainBuffer[cellIndex] };
	const float height{ terrain.x + terrain.y };

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
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
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

__device__ __inline__ int linear_block_10x10(int x, int y) { return (y + 1) * 10 + (x + 1); }

template <bool TUseAvalancheStrength>
__global__ void sharedAtomicInPlaceAvalanchingKernel(Buffer<float2> t_terrainBuffer)
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
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
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

void avalanching(const LaunchParameters& t_launchParameters)
{
	switch (t_launchParameters.avalancheMode)
	{
	case AvalancheMode::Atomic:
		setupAtomicAvalanchingKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);

	    for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	    {
			atomicAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);
		    finishAtomicAvalanchingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);
	    } 

	    break;
	case AvalancheMode::AtomicInPlace:
		Buffer<float2> terrainBuffer{ reinterpret_cast<Buffer<float2>>(t_launchParameters.tmpBuffer) };

		setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		// Todo: Magic constants (5 and 10) - might want to make these named
		for (int i = 0; i < (t_launchParameters.avalancheIterations - 5); ++i)
		{
			if (i % 10 == 0)
			{
				atomicInPlaceAvalanchingKernel<true><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
			}
			else
			{
				sharedAtomicInPlaceAvalanchingKernel<false><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
			}
		}

		for (int i = t_launchParameters.avalancheIterations - 5; i < (t_launchParameters.avalancheIterations); ++i)
		{
			sharedAtomicInPlaceAvalanchingKernel<true><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
		}

		finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		break;
	}
}

}
