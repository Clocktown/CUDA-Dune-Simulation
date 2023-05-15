#include "simulation_parameters.hpp"
#include <dunes/device/constants.cuh>
#include <sthe/config/debug.hpp>

namespace dunes
{

void upload(const SimulationParameters& t_simulationParameters)
{
	CU_CHECK_ERROR(cudaMemcpyToSymbol(c_parameters, &t_simulationParameters, sizeof(SimulationParameters)));
}

}
