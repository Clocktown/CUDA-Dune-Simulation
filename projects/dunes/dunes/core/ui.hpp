#pragma once

#include <sthe/sthe.hpp>

namespace dunes
{

class UI : public sthe::Component
{
public:
	// Constructors
	UI();
	UI(const UI& t_ui) = default;
	UI(UI&& t_ui) = default;

	// Destructor
	~UI() = default;

	// Operators
	UI& operator=(const UI& t_ui) = default;
	UI& operator=(UI&& t_ui) = default;

	// Functionality
	void onGUI();
private:

};

}
