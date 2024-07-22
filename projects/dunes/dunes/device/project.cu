#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "multigrid.cuh"
#include <sthe/config/debug.hpp>
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace dunes {

	__global__ void initDivergencePressureKernel(const Array2D<float2> t_windArray, Buffer<float> t_divergenceBuffer, Buffer<float> t_pressureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		t_pressureBuffer[cellIndex] = 0.f;

		const float divergence = -0.5f * (
				(t_windArray.read(getWrappedCell(cell + c_offsets[0])).x - t_windArray.read(getWrappedCell(cell + c_offsets[4])).x) +
				(t_windArray.read(getWrappedCell(cell + c_offsets[2])).y - t_windArray.read(getWrappedCell(cell + c_offsets[6])).y)
			);

		t_divergenceBuffer[cellIndex] = divergence;
	}

	__global__ void projectKernel(const Array2D<float4> t_resistanceArray, const Buffer<float> t_divergenceBuffer, const Buffer<float> t_pressureABuffer, Buffer<float> t_pressureBBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		float new_pressure = t_divergenceBuffer[cellIndex];
		for (int i = 0; i < 8; i += 2) {
			const int2 nextCell = getWrappedCell(cell + c_offsets[i]);
			const int nextCellIndex = getCellIndex(nextCell);

			new_pressure += t_pressureABuffer[nextCellIndex];
		}
		new_pressure *= 0.25f;
		new_pressure *= (1.f - t_resistanceArray.read(cell).x);

		t_pressureBBuffer[cellIndex] = new_pressure;
	}

	__global__ void finalizeVelocities(Array2D<float4> t_resistanceArray, Array2D<float2> t_windArray, const Buffer<float> t_pressureBuffer) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		float2 velocity = t_windArray.read(cell);

		velocity.x -= 0.5f * (
				t_pressureBuffer[getCellIndex(getWrappedCell(cell + c_offsets[0]))] 
			-	t_pressureBuffer[getCellIndex(getWrappedCell(cell + c_offsets[4]))]
			);
		velocity.y -= 0.5f * (
				0.5f * (t_pressureBuffer[getCellIndex(getWrappedCell(cell + c_offsets[2]))] 
			-	t_pressureBuffer[getCellIndex(getWrappedCell(cell + c_offsets[6]))])
			);

		t_windArray.write(cell, velocity);
	}

	__global__ void multiplyWindShadowKernel(Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		float2 velocity = t_windArray.read(cell) * (1.f - t_resistanceArray.read(cell).x);
		float4 resistance = t_resistanceArray.read(cell);
		resistance.x = 0.0f;
		
		t_windArray.write(cell, velocity);
		t_resistanceArray.write(cell, resistance);
	}

	__global__ void setupProjection(const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> velocityBufferX, Buffer<float> velocityBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		const float2 velocity = t_windArray.read(cell);
		float4 resistance = t_resistanceArray.read(cell);
		
		velocityBufferX[cellIndex] = velocity.x * (1.0f - resistance.x);
		velocityBufferY[cellIndex] = velocity.y * (1.0f - resistance.x);

		resistance.x = 0.0f;
		t_resistanceArray.write(cell, resistance);
	}

	__global__ void fftProjection(Buffer<cuComplex> frequencyBufferX, Buffer<cuComplex> frequencyBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		cuComplex freqX{ frequencyBufferX[cellIndex] };
		cuComplex freqY{ frequencyBufferY[cellIndex] };

		const int iix{ cell.x };
		const int iiy{ cell.y > c_parameters.gridSize.y / 2 ? cell.y - c_parameters.gridSize.y : cell.y };

		const float kk{ static_cast<float>(iix * iix + iiy * iiy) };
		float diff = 1.0f / (1.0f + kk * 0.0025f * c_parameters.deltaTime);

		if (kk > 0.0f)
		{
			const float rkk{ 1.0f / kk };
			const float rkp{ iix * freqX.x + iiy * freqY.x };
			const float ikp{ iix * freqX.y + iiy * freqY.y };

			freqX.x -= rkk * rkp * iix;
			freqX.y -= rkk * ikp * iix;
			freqY.x -= rkk * rkp * iiy;
			freqY.y -= rkk * ikp * iiy;
		}

		frequencyBufferX[cellIndex] = freqX;
		frequencyBufferY[cellIndex] = freqY;
	}

	__global__ void finalizeProjection(Array2D<float2> t_windArray, Buffer<float> velocityBufferX, Buffer<float> velocityBufferY)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };
		const float scale{ 1.0f / static_cast<float>(c_parameters.gridSize.x * c_parameters.gridSize.y) };

		const float2 velocity{ velocityBufferX[cellIndex], velocityBufferY[cellIndex] };
		t_windArray.write(cell, scale * velocity);
	}

	// Debug Operator for divergence reduction
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
		Buffer<float> divergenceBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };
		Buffer<float> pressureABuffer{ t_launchParameters.tmpBuffer + 0 * t_simulationParameters.cellCount };
		Buffer<float> pressureBBuffer{ t_launchParameters.tmpBuffer + 1 * t_simulationParameters.cellCount };

		if (t_launchParameters.projection.mode == ProjectionMode::Jacobi)
		{
			multiplyWindShadowKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.resistanceArray);
			initDivergencePressureKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, divergenceBuffer, pressureABuffer);

			// Debug
			float div = thrust::transform_reduce(thrust::device, divergenceBuffer, divergenceBuffer + t_simulationParameters.cellCount, Unary(), 0.0f, Binary());
			printf("%f -> ", div / t_simulationParameters.cellCount);

			for (int i = 0; i < t_launchParameters.projection.jacobiIterations; ++i) 
			{
		        projectKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, divergenceBuffer, pressureABuffer, pressureBBuffer);
		        std::swap(pressureABuffer, pressureBBuffer);
		    }

		    finalizeVelocities<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, t_launchParameters.windArray, pressureABuffer);
		 	
			// Debug
		    initDivergencePressureKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, divergenceBuffer, pressureABuffer);
		    div = thrust::transform_reduce(thrust::device, divergenceBuffer, divergenceBuffer + t_simulationParameters.cellCount, Unary(), 0.0f, Binary());
		    printf("%f\n", div / t_simulationParameters.cellCount);
		}
		else if (t_launchParameters.projection.mode == ProjectionMode::FFT)
		{
			// Debug
			initDivergencePressureKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, divergenceBuffer, pressureABuffer);
            float div = thrust::transform_reduce(thrust::device, divergenceBuffer, divergenceBuffer + t_simulationParameters.cellCount, Unary(), 0.0f, Binary());
            printf("%f -> ", div / t_simulationParameters.cellCount);

			setupProjection<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.projection.velocities[0], t_launchParameters.projection.velocities[1]);

		    CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.projection.planR2C, t_launchParameters.projection.velocities[0], t_launchParameters.projection.frequencies[0]));
		    CUFFT_CHECK_ERROR(cufftExecR2C(t_launchParameters.projection.planR2C, t_launchParameters.projection.velocities[1], t_launchParameters.projection.frequencies[1]));

		    fftProjection<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.projection.frequencies[0], t_launchParameters.projection.frequencies[1]);

		    CUFFT_CHECK_ERROR(cufftExecC2R(t_launchParameters.projection.planC2R, t_launchParameters.projection.frequencies[0], t_launchParameters.projection.velocities[0]));
		    CUFFT_CHECK_ERROR(cufftExecC2R(t_launchParameters.projection.planC2R, t_launchParameters.projection.frequencies[1], t_launchParameters.projection.velocities[1]));

		    finalizeProjection<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.projection.velocities[0], t_launchParameters.projection.velocities[1]);
		
			// Debug
		    initDivergencePressureKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, divergenceBuffer, pressureABuffer);
		    div = thrust::transform_reduce(thrust::device, divergenceBuffer, divergenceBuffer + t_simulationParameters.cellCount, Unary(), 0.0f, Binary());
		    printf("%f\n", div / t_simulationParameters.cellCount);
		}
	}
}