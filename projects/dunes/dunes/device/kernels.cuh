#pragma once

#include <dunes/core/launch_parameters.hpp>
#include <cuda_runtime.h>

namespace dunes
{

void initialization(const LaunchParameters& t_launchParameters);
void venturi(const LaunchParameters& t_launchParameters);
void windShadow(const LaunchParameters& t_launchParameters);
void saltation(const LaunchParameters& t_launchParameters);
void reptation(const LaunchParameters& t_launchParameters);
void avalanching(const LaunchParameters& t_launchParameters);

}
