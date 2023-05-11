#pragma once

#include "simulation_parameter.cuh"


namespace dunes
{

extern __constant__ dunes::SimulationParameter c_simulationParameter;

extern __constant__ int2 c_offsets[8];

extern __constant__ float c_rDistances[8];

}
