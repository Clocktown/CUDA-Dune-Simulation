#pragma once

#include "simulation_parameters.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <array>
#include <vector>

namespace dunes
{

enum class TimeMode : unsigned char
{
	DeltaTime, FixedDeltaTime
};

enum class SaltationMode : unsigned char
{
	Backward, Forward
};

enum class WindWarpingMode : unsigned char
{
	None, Standard
};

enum class WindShadowMode : unsigned char
{
	Linear, Curved
};

enum class AvalancheMode : unsigned char
{
	AtomicBuffered, AtomicInPlace, SharedAtomicInPlace, MixedInPlace, Multigrid, Taylor, Jacobi
};

enum class BedrockAvalancheMode : unsigned char
{
	ToSand, ToBedrock
};

struct LaunchParameters
{
	unsigned int blockSize1D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	dim3 gridSize2D;

	unsigned int optimalBlockSize1D;
	dim3 optimalBlockSize2D;
	unsigned int optimalGridSize1D;
	dim3 optimalGridSize2D;

	SaltationMode saltationMode{ SaltationMode::Forward };
	WindWarpingMode windWarpingMode{ WindWarpingMode::None };
	WindShadowMode windShadowMode{ WindShadowMode::Linear };
	AvalancheMode avalancheMode{ AvalancheMode::AtomicInPlace };
	BedrockAvalancheMode bedrockAvalancheMode{ BedrockAvalancheMode::ToSand };
	bool useBilinear{ true };
	int avalancheIterations{ 50 };
	int pressureProjectionIterations{ 0 };
	int bedrockAvalancheIterations{ 2 };
	int avalancheSoftIterationModulus{ 10 };
	int avalancheFinalSoftIterations{ 5 };
	int multigridLevelCount{ 1 };
	int multigridVCycleIterations{ 1 };
	int multigridSolverIterations{ 50 };
	TimeMode timeMode{ TimeMode::DeltaTime };

	Array2D<float2> terrainArray;
	Array2D<float2> windArray;
	Array2D<float4> resistanceArray; // .x = wind shadow, .y = vegetation, .z = erosion, .w = sticky
	Buffer<float> slabBuffer;
	Buffer<float> tmpBuffer; // 4 * gridSize.x * gridSize.y
	WindWarping windWarping;
	std::vector<MultigridLevel> multigrid;

	cufftHandle fftPlan;
};

struct NoiseGenerationParameters 
{
	float2 offset{ 0.f, 0.f };
	float2 stretch{ 1.f, 1.f };
	float2 border{ 0.1f , 0.1f };
    float scale = 100.f;
    float bias = 0.f;
    int iters = 0;
	bool flat = false;
	bool enabled = true;
	bool uniform_random = false;
};

constexpr int NumNoiseGenerationTargets = 4;

enum class NoiseGenerationTarget : unsigned char
{
	Bedrock, Sand, Vegetation, AbrasionResistance
};

struct InitializationParameters
{
	NoiseGenerationParameters noiseGenerationParameters[NumNoiseGenerationTargets]{
		{},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 100.f, 10.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false},
		{{ 0.f, 0.f }, { 1.f, 1.f }, { 0.1f , 0.1f }, 1.f, 0.f, 0, true, true, false}
	};
};

}
