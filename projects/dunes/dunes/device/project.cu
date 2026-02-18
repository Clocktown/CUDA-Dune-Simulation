#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <sthe/config/debug.hpp>
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace dunes {
	__global__ void setupProjection(const Array2D<half2> t_windArray, Buffer<float> velocityBufferX, Buffer<float> velocityBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int width{ 2 * (c_parameters.gridSize.x / 2 + 1) };
		const int cellIndex{ cell.x + cell.y * width };

		const float2 velocity = __half22float2(t_windArray.read(cell));
		velocityBufferX[cellIndex] = velocity.x;
		velocityBufferY[cellIndex] = velocity.y;
	}

	__global__ void fftProjection(Buffer<cuComplex> frequencyBufferX, Buffer<cuComplex> frequencyBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };
		const int2 size{ c_parameters.gridSize.x / 2 + 1, c_parameters.gridSize.y };

		if (isOutside(cell, size))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell, size) };

		cuComplex xterm{ frequencyBufferX[cellIndex] };
		cuComplex yterm{ frequencyBufferY[cellIndex] };

		const int iix{ cell.x };
		const int iiy{ cell.y > size.y / 2 ? cell.y - size.y : cell.y };

		const float kk{ static_cast<float>(iix * iix + iiy * iiy) };

		constexpr float viscosity{ 0.0f };
		const float diffusion = 1.0f / (1.0f + kk * viscosity * c_parameters.deltaTime);
		xterm.x *= diffusion;
		xterm.y *= diffusion;
		yterm.x *= diffusion;
		yterm.y *= diffusion;

		if (kk > 0.0f)
		{
			const float rkk{ 1.0f / kk };
			const float rkp{ iix * xterm.x + iiy * yterm.x };
			const float ikp{ iix * xterm.y + iiy * yterm.y };

			xterm.x -= rkk * rkp * iix;
			xterm.y -= rkk * ikp * iix;
			yterm.x -= rkk * rkp * iiy;
			yterm.y -= rkk * ikp * iiy;
		}

		frequencyBufferX[cellIndex] = xterm;
		frequencyBufferY[cellIndex] = yterm;
	}

	__global__ void finalizeProjection(Array2D<half2> t_windArray, Buffer<float> velocityBufferX, Buffer<float> velocityBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int width{ 2 * (c_parameters.gridSize.x / 2 + 1) };

		const int cellIndex{ cell.x + cell.y * width };
		const float scale{ 1.0f / static_cast<float>(c_parameters.gridSize.x * c_parameters.gridSize.y) };

		const float2 velocity{ velocityBufferX[cellIndex], velocityBufferY[cellIndex] };
		t_windArray.write(cell, __float22half2_rn(scale * velocity));
	}

	// Debug Operators for divergence reduction
	struct Unary
	{
		__device__ float operator()(float x)
		{
			return fabsf(x);
		}
	};
	struct Binary
	{
		__device__ float operator()(float x, float y)
		{
			return x + y;
		}
	};

	void pressureProjection(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) 
	{
		if (t_launchParameters.projection.mode == ProjectionMode::Jacobi)
		{
			// Removed
		}
		else if (t_launchParameters.projection.mode == ProjectionMode::FFT)
		{
			setupProjection<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.projection.velocities[0], t_launchParameters.projection.velocities[1]);

		    CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.projection.planR2C, (cufftReal*)t_launchParameters.projection.velocities[0], (cuComplex*)t_launchParameters.projection.velocities[0]));
		    CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.projection.planR2C, (cufftReal*)t_launchParameters.projection.velocities[1], (cuComplex*)t_launchParameters.projection.velocities[1]));

			dim3 gridSize;
			gridSize.x = static_cast<unsigned int>(ceilf(static_cast<float>(t_simulationParameters.gridSize.x / 2 + 1) / 8.0f));
			gridSize.y = static_cast<unsigned int>(ceilf(static_cast<float>(t_simulationParameters.gridSize.y) / 8.0f));
			gridSize.z = 1;

		    fftProjection<<<gridSize, dim3{ 8, 8, 1 } >> >((cuComplex*)t_launchParameters.projection.velocities[0], (cuComplex*)t_launchParameters.projection.velocities[1]);

		    CUFFT_CHECK_ERROR(cufftExecC2R(t_launchParameters.projection.planC2R, (cuComplex*)t_launchParameters.projection.velocities[0], (cufftReal*)t_launchParameters.projection.velocities[0]));
		    CUFFT_CHECK_ERROR(cufftExecC2R(t_launchParameters.projection.planC2R, (cuComplex*)t_launchParameters.projection.velocities[1], (cufftReal*)t_launchParameters.projection.velocities[1]));

		    finalizeProjection<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.projection.velocities[0], t_launchParameters.projection.velocities[1]);
		
			// Debug
		    //initDivergencePressureKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, divergenceBuffer, pressureABuffer);
		    //div = thrust::transform_reduce(thrust::device, divergenceBuffer, divergenceBuffer + t_simulationParameters.cellCount, Unary(), 0.0f, Binary());
		    //printf("%f\n", div / t_simulationParameters.cellCount);
		}
	}
}