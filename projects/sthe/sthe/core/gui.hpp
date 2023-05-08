#pragma once

#include <imgui.h>

namespace sthe
{

class GUI
{
public:
	// Constructors
	GUI(const GUI& t_gui) = delete;
	GUI(GUI&& t_gui) = delete;

	// Destructor
	~GUI();

	// Functionality
	void start();
	void render();

	// Operators
	GUI& operator=(const GUI& t_gui) = delete;
	GUI& operator=(GUI&& t_gui) = delete;
private:
	// Constructor
	GUI();

	// Friends
	friend GUI& getGUI();
};

GUI& getGUI();

}
