#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainBuffer[cellIndex] = t_terrainArray.read(cell);
}

__global__ void finishAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainArray.write(cell, t_terrainBuffer[cellIndex]);
}

__global__ void avalanchingKernel(Buffer<float2> t_terrainBuffer)
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
		const float avalancheSize{ fminf(c_parameters.avalancheStrength * maxAvalanche /
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

__global__ void avalanchingKernelShared8x8(Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	__shared__ float s[10 * 10];
	int2 threadID = int2 {int(threadIdx.x), int(threadIdx.y)};
    int              threadIndx = threadIdx.y * blockDim.x + threadIdx.x;
    int              idx        = cellIndex;
    int              idx_shared = linear_block_10x10(threadIdx.x, threadIdx.y);
    float2            zp         = t_terrainBuffer[idx];
    s[idx_shared]               = zp.x + zp.y;
    int2 off {0, 0};
    if(threadIndx < 10)
    { // leftmost column
        off = int2 {-1, threadIndx - 1};
    }
    else if(threadIndx < 20)
    { // rightmost column
        off = int2 {8, threadIndx - 11};
    }
    else if(threadIndx < 28)
    { // top row (minus edge columns)
        off = int2 {threadIndx - 20, 8};
    }
    else if(threadIndx < 36)
    { // bottom row (minus edge columns)
        off = int2 {threadIndx - 28, -1};
    }
    if(threadIndx < 36)
    {
		int2 b = cell + off;
        b = getWrappedCell(b);
		float2 v = t_terrainBuffer[getCellIndex(b)];
        s[linear_block_10x10(off.x, off.y)] = v.x + v.y;
    }
    __syncthreads();

	//const float2 terrain{ zp };
	const float height{ zp.x + zp.y };

	//int nextCellIndices[8];
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
    if(threadIndx < 36)
    {
		s[linear_block_10x10(off.x, off.y)] = 0.f;
    }
	__syncthreads();

	if (avalancheSum > 1e-6f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf(c_parameters.avalancheStrength * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), zp.y) };


		const float scale{ avalancheSize * rAvalancheSum };

		for(int i = 0; i < 8; ++i)
        {
            if(avalanches[i] <= 0)
                continue;
            avalanches[i] *= scale;
            // slopes[i] *= sand_to_move;
            int2 b = threadID + c_offsets[i];
            atomicAdd(&s[linear_block_10x10(b.x, b.y)], avalanches[i]);
        }
        atomicAdd(&s[idx_shared], -avalancheSize);
	}

	__syncthreads();
	if((threadID.x % 7) == 0 || (threadID.y % 7) == 0)
	{ // border, need atomic add
		atomicAdd(&t_terrainBuffer[idx].y, s[idx_shared]);
	}
	else
	{
		t_terrainBuffer[idx].y = zp.y + s[idx_shared];
	}
	// ps[idx] = zp + s[idx_shared];
	if(threadIndx < 36)
	{
		int2 b = cell + off;
		b = getWrappedCell(b);
		atomicAdd(&t_terrainBuffer[getCellIndex(b)].y, s[linear_block_10x10(off.x, off.y)]);
	}
}

void avalanching(const LaunchParameters& t_launchParameters)
{
	Buffer<float2> terrainBuffer{ reinterpret_cast<Buffer<float2>>(t_launchParameters.tmpBuffer) };

	setupAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);
	
	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		avalanchingKernelShared8x8<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(terrainBuffer);
	}

	finishAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);
}

}
