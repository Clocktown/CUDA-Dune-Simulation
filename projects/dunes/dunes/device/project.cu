#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "multigrid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

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
		float4 resistance = t_resistanceArray.read(cell);
		resistance.x = 0.f;
		t_resistanceArray.write(cell, resistance);

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

	__global__ void multiplyWindShadowKernel(Array2D<float2> t_windArray, const Array2D<float4> t_resistanceArray) {
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const int cellIndex{ getCellIndex(cell) };

		float2 velocity = t_windArray.read(cell) * (1.f - t_resistanceArray.read(cell).x);

		t_windArray.write(cell, velocity);
	}



	void pressureProjection(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters) {
		if (t_launchParameters.pressureProjectionIterations <= 0) {
			return;
		}
		windShadow(t_launchParameters);
		Buffer<float> pressureABuffer{ t_launchParameters.tmpBuffer + 0 * t_simulationParameters.cellCount };
		Buffer<float> pressureBBuffer{ t_launchParameters.tmpBuffer + 1 * t_simulationParameters.cellCount };
		Buffer<float> divergenceBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };

		multiplyWindShadowKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, t_launchParameters.resistanceArray);

		initDivergencePressureKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.windArray, divergenceBuffer, pressureABuffer);

		for (int i = 0; i < t_launchParameters.pressureProjectionIterations; ++i) {
			projectKernel<< <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, divergenceBuffer, pressureABuffer, pressureBBuffer);
			std::swap(pressureABuffer, pressureBBuffer);
		}
		finalizeVelocities << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.resistanceArray, t_launchParameters.windArray, pressureABuffer);

	}
}