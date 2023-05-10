#include "simulation_parameter.cuh"
#include "constant.cuh"
#include <sthe/config/debug.hpp>

namespace dunes
{

void upload(const SimulationParameter& t_simulationParameter)
{
	CU_CHECK_ERROR(cudaMemcpyToSymbol(c_simulationParameter, &t_simulationParameter, sizeof(SimulationParameter)));
}

}
