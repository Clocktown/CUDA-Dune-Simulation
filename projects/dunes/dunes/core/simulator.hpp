#pragma once

#include <sthe/sthe.hpp>

namespace dunes
{

class Simulator : sthe::Component
{
public:
	// Constructors
	Simulator();
	Simulator(const Simulator& t_simulator) = default;
	Simulator(Simulator&& t_simulator) = default;

	// Destructor
	~Simulator() = default;

	// Operators
	Simulator& operator=(const Simulator& t_simulator) = default;
	Simulator& operator=(Simulator&& t_simulator) = default;

	// Functionality
	void awake();
	void update();
	void onGUI();
};

}
