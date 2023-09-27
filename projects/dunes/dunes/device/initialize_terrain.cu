#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{
#define M_PI 3.1415926535897932384626433832795

	__device__ float frand(float2 c) { return fract(sin(dot(c, float2{ 12.9898, 78.233 })) * 43758.5453); }

	__device__ float noise(float2 p, float freq)
	{
		float2 coords = p * freq;
		float2  ij = floorf(coords);
		float2  xy = coords - ij;

		//xy = 3.*xy*xy-2.*xy*xy*xy; // Alternative to cos
		xy = 0.5f * (1.f - cos(M_PI * xy));
		float a = frand((ij + float2{ 0., 0. }));
		float b = frand((ij + float2{ 1., 0. }));
		float c = frand((ij + float2{ 0., 1. }));
		float d = frand((ij + float2{ 1., 1. }));
		return bilerp(a, b, c, d, xy.x, xy.y);
	}

	__device__ float pNoise(float2 p, int res)
	{
		float persistance = .5;
		float n = 0.;
		float normK = 0.;
		float f = 4.;
		float amp = 1.;
		for (int i = 0; i <= res; i++)
		{
			n += amp * noise(p, f);
			f *= 2.;
			normK += amp;
			amp *= persistance;
		}
		float nf = n / normK;
		return nf * nf * nf * nf;
	}

	__device__ float seamless_pNoise(float2 off, float2 stretch, float2 uv, int res, float2 border)
	{
		float2 centered_uv_xy = 2.f * (uv - float2{ 0.5f, 0.5f });
		float2  limits = float2{ 1.f, 1.f } - 2.f * border;
		float2 distance = float2{ 0.f, 0.f };

		distance = max(abs(centered_uv_xy) - limits, float2{ 0.f, 0.f });

		centered_uv_xy = -1.f * sign(centered_uv_xy) * (limits + distance);

		distance /= 2.f * border;

		float2 xy_uv = 0.5f * centered_uv_xy + float2{ 0.5f, 0.5f };
		xy_uv = off + stretch * xy_uv;
		float2 base_uv = off + stretch * uv;

		float base_sample = pNoise(base_uv, res);

		if ((distance.x <= 0.f) && (distance.y <= 0.f))
		{
			return base_sample;
		}

		return bilerp(
			base_sample, 
			pNoise(float2{ xy_uv.x, base_uv.y }, res),
			pNoise(float2{ base_uv.x, xy_uv.y }, res),
			pNoise(xy_uv, res),
			0.5f * smoothstep(0.f, 1.f, distance.x), 
			0.5f * smoothstep(0.f, 1.f, distance.y)
		);
	}

	__global__ void initializeTerrainKernel(Array2D<float2> t_terrainArray, Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, InitializationParameters t_initializationParameters)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		const float2 uv = (make_float2(cell) + 0.5f) / make_float2(c_parameters.gridSize);

		const float2 curr_terrain = t_terrainArray.read(cell);
		const float4 curr_resistance = t_resistanceArray.read(cell);

		const int indices[4]{
			(int)NoiseGenerationTarget::Bedrock,
			(int)NoiseGenerationTarget::Sand,
			(int)NoiseGenerationTarget::Vegetation,
			(int)NoiseGenerationTarget::AbrasionResistance
		};
		float values[4] = { curr_terrain.x, curr_terrain.y, curr_resistance.y, curr_resistance.z };

		for (int i = 0; i < 4; ++i) {
			const auto& params = t_initializationParameters.noiseGenerationParameters[indices[i]];
			values[i] = params.enabled ?
				(params.bias +
					(params.uniform_random ? params.scale * frand(params.offset + params.stretch * uv) :
					(params.flat ?
						0.0f :
						params.scale * seamless_pNoise(params.offset, params.stretch, uv, params.iters, params.border)))) :
				values[i];
		}

		float2 center{ make_float2(c_parameters.gridSize) / 2.f };

		// Wind Tunnel initialization
		/*center.x -= 100;
		const float2 cellf{ make_float2(cell) };
		if (length(cellf - center) <= 150 || ((cellf.x > center.x) && (cellf.x - 400 < center.x) && (abs(cellf.y - center.y) <= 150)))
		{
			const float2 terrain{ 100.0f, 0.0f };
			t_terrainArray.write(cell, terrain);
		}
		else
		{
			const float2 terrain{ 0.f, fmaxf(values[1], 0.f)};
			t_terrainArray.write(cell, terrain);
		}*/

		// Sand Column initialization
		/*const float2 cellf{ make_float2(cell) };
		if (length(cellf-center) < 100.f)
		{
			const float2 terrain{ 0.f, 402.f };
			t_terrainArray.write(cell, terrain);
		}
		else
		{
			const float2 terrain{ 0.f, 2.f};
			t_terrainArray.write(cell, terrain);
		}*/

		// Regular initialization
		const float2 terrain{ values[0], fmaxf(values[1], 0.f)};
		t_terrainArray.write(cell, terrain);

		const float4 resistance{ 0.0f, clamp(values[2], 0.f, 1.f), clamp(values[3], 0.f, 1.f), 0.0f};
		t_resistanceArray.write(cell, resistance);

		t_slabBuffer[getCellIndex(cell)] = 0.0f;
	}

	__global__ void addSandForCoverageKernel(Array2D<float2> t_terrainArray, float amount)
	{
		const int2 cell{ getGlobalIndex2D() };

		if (isOutside(cell))
		{
			return;
		}

		float2 curr_terrain = t_terrainArray.read(cell);

		curr_terrain.y += frand(make_float2(cell)) * 2.f * amount;
		curr_terrain.y = fmaxf(curr_terrain.y, 0.f);

		t_terrainArray.write(cell, curr_terrain);
	}

	void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters)
	{
		initializeTerrainKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_initializationParameters);
	}

	void addSandForCoverage(const LaunchParameters& t_launchParameters, float amount) {
		addSandForCoverageKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, amount);
	}

}
