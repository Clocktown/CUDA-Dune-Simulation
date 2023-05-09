#pragma once

#include <sthe/sthe.hpp>
#include <dunes/device/simulation.cuh>

namespace dunes
{

class Simulation : public sthe::Component
{
public:
	// Constructors
	Simulation();
	Simulation(const Simulation& t_simulation) = default;
	Simulation(Simulation&& t_simulation) = default;

	// Destructor
	~Simulation() = default;

	// Operators
	Simulation& operator=(const Simulation& t_simulation) = default;
	Simulation& operator=(Simulation&& t_simulation) = default;

	// Functionality
	void awake();
	void update();
	void onGUI();
private:
	// Attributes
	device::Simulation m_data;
};

}
