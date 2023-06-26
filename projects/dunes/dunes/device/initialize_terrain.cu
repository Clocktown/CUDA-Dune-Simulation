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
    float unit = 1.f / freq;
    float2 coords = p / unit;
    float2  ij   = floorf(coords);
    float2  xy = mod(p, unit) / unit;
    
    // xy = 3.*xy*xy-2.*xy*xy*xy;
    xy       = 0.5f * (1.f - cos(M_PI * xy));
    float a = frand((ij + float2{ 0., 0. }));
    float b = frand((ij + float2{ 1., 0. }));
    float c = frand((ij + float2{ 0., 1. }));
    float d = frand((ij + float2{ 1., 1. }));
    float x1 = lerp(a, b, xy.x);
    float x2 = lerp(c, d, xy.x);
    return lerp(x1, x2, xy.y);
}

__device__ float pNoise(float2 p, int res)
{
    float persistance = .5;
    float n           = 0.;
    float normK       = 0.;
    float f           = 4.;
    float amp         = 1.;
    for(int i = 0; i <= res; i++)
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
    float2  centered_uv = 2.f * (uv - float2{ 0.5f, 0.5f });
    float2  centered_uv_x = 2.f * (uv - float2{ 0.5f, 0.5f });
    float2  centered_uv_y = 2.f * (uv - float2{ 0.5f, 0.5f });
    float2  limits = float2{ 1.f, 1.f } - 2.f * border;
    float distance_from_border_x = 0.f;
    float distance_from_border_y = 0.f;

    if(centered_uv.x < -limits.x)
    {
        float d         = -limits.x - centered_uv.x;
        centered_uv.x   = limits.x + d;
        centered_uv_x.x = limits.x + d;
        d /= 2.f * border.x;
        distance_from_border_x = d;
    }
    else if(centered_uv.x > limits.x)
    {
        float d         = centered_uv.x - limits.x;
        centered_uv.x   = -limits.x - d;
        centered_uv_x.x = -limits.x - d;
        d /= 2.f * border.x;
        distance_from_border_x = d;
    }

    if(centered_uv.y < -limits.y)
    {
        float d         = -limits.y - centered_uv.y;
        centered_uv.y   = limits.y + d;
        centered_uv_y.y = limits.y + d;
        d /= 2.f * border.y;
        distance_from_border_y = d;
    }
    else if(centered_uv.y > limits.y)
    {
        float d         = centered_uv.y - limits.y;
        centered_uv.y   = -limits.y - d;
        centered_uv_y.y = -limits.y - d;
        d /= 2.f * border.y;
        distance_from_border_y = d;
    }

    float2 xy_uv = 0.5f * centered_uv + float2{ 0.5f, 0.5f };
    float2 x_uv = 0.5f * centered_uv_x + float2{ 0.5f, 0.5f };
    float2 y_uv = 0.5f * centered_uv_y + float2{ 0.5f, 0.5f };
    xy_uv         = off + stretch * xy_uv;
    x_uv          = off + stretch * x_uv;
    y_uv          = off + stretch * y_uv;
    float2 first_uv = off + stretch * uv;

    float sample_a = pNoise(first_uv, res);

    if((distance_from_border_x <= 0.f) && (distance_from_border_y <= 0.f))
    {
        return sample_a;
    }

    float sample_b      = pNoise(y_uv, res);
    float second_sample = pNoise(x_uv, res);
    float third_sample  = pNoise(xy_uv, res);

    sample_a = lerp(sample_a, second_sample, 0.5f * smoothstep(0.f, 1.f, distance_from_border_x));
    sample_b = lerp(sample_b, third_sample, 0.5f * smoothstep(0.f, 1.f, distance_from_border_x));

    sample_a = lerp(sample_a, sample_b, 0.5f * smoothstep(0.f, 1.f, distance_from_border_y));

    return sample_a;
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

    auto bedrockParams = t_initializationParameters.noiseGenerationParameters[(int)NoiseGenerationTarget::Bedrock];
    auto sandParams = t_initializationParameters.noiseGenerationParameters[(int)NoiseGenerationTarget::Sand];
    auto vegetationParams = t_initializationParameters.noiseGenerationParameters[(int)NoiseGenerationTarget::Vegetation];
    auto abrasionParams = t_initializationParameters.noiseGenerationParameters[(int)NoiseGenerationTarget::AbrasionResistance];

    const float bedrockHeight{
        bedrockParams.enabled ?
            (bedrockParams.bias +
            (bedrockParams.flat ?
                0.0f :
                bedrockParams.scale * seamless_pNoise(bedrockParams.offset, bedrockParams.stretch, uv, bedrockParams.iters, bedrockParams.border))) :
            curr_terrain.x
    };

    const float sandHeight{
        sandParams.enabled ?
            (sandParams.bias +
            (sandParams.flat ?
                0.0f :
                sandParams.scale * seamless_pNoise(sandParams.offset, sandParams.stretch, uv, sandParams.iters, sandParams.border))) :
            curr_terrain.y
    };

    const float vegetationRes{
        vegetationParams.enabled ?
            (vegetationParams.bias +
            (vegetationParams.flat ?
                0.0f :
                vegetationParams.scale * seamless_pNoise(vegetationParams.offset, vegetationParams.stretch, uv, vegetationParams.iters, vegetationParams.border))) :
            curr_resistance.y
    };

    const float abrasionRes{
        abrasionParams.enabled ?
            (abrasionParams.bias +
            (abrasionParams.flat ?
                0.0f :
                abrasionParams.scale * seamless_pNoise(abrasionParams.offset, abrasionParams.stretch, uv, abrasionParams.iters, abrasionParams.border))) :
            curr_resistance.z
    };
	
	const float2 terrain{ bedrockHeight, fmaxf(sandHeight, 0.f) };
	t_terrainArray.write(cell, terrain);
	
	const float4 resistance{ 0.0f, clamp(vegetationRes, 0.f, 1.f), clamp(abrasionRes, 0.f, 1.f), 0.0f };
	t_resistanceArray.write(cell, resistance);

	t_slabBuffer[getCellIndex(cell)] = 0.0f;
}

void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters)
{
	initializeTerrainKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_initializationParameters);
}

}
