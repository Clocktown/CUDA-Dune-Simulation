#include "simulation_parameter.cuh"

namespace dunes
{

#define RSQRT2 0.707106782f

__constant__ SimulationParameter c_simulationParameter {};

__constant__ int2 c_offsets[8] {int2 {1, 0},
                                       int2 {1, 1},
                                       int2 {0, 1},
                                       int2 {-1, 1},
                                       int2 {-1, 0},
                                       int2 {-1, -1},
                                       int2 {0, -1},
                                       int2 {1, -1}};

__constant__ float c_rDistances[8] {1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2, 1.0f, RSQRT2};

#undef RSQRT2

} // namespace dunes