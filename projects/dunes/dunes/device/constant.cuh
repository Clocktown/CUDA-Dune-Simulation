#pragma once

#include "simulation.cuh"

namespace dunes
{
namespace device
{

__constant__ Simulation t_simulation;

__constant__ int2 t_offsets[8]{ int2{ 1, 0 }, int2{ 1, 1 }, int2{ 0, 1 }, int2{ -1, 1 },
                                int2{ -1, 0 }, int2{ -1, -1 }, int2{ 0, -1 }, int2{ 1, -1 } };

__constant__ float t_distances[8]{ 1.0f, sqrtf(2.0f), 1.0f, sqrtf(2.0f), 
                                   1.0f, sqrtf(2.0f), 1.0f, sqrtf(2.0f) };


}
}
