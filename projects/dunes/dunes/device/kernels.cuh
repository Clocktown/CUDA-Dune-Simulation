#pragma once

#include <dunes/core/launch_parameters.hpp>
#include <cuda_runtime.h>

namespace dunes
{

void initializeTerrain(const LaunchParameters& t_launchParameters, const InitializationParameters& t_initializationParameters);
void addSandForCoverage(const LaunchParameters& t_launchParameters, float amount);
void initializeWindWarping(const LaunchParameters& t_launchParameters);

void venturi(const LaunchParameters& t_launchParameters);
void windWarping(const LaunchParameters& t_launchParameters);
void windShadow(const LaunchParameters& t_launchParameters);
void sticky(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters);
void saltation(const LaunchParameters& t_launchParameters);
void reptation(const LaunchParameters& t_launchParameters);
void avalanching(const LaunchParameters& t_launchParameters);

float coverage(const LaunchParameters& t_launchParameters, unsigned int* coverageMap, int num_cells, float threshold);

}
