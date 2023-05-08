#pragma once

#include <chrono>

namespace sthe
{

class Stopwatch
{
public:
	// Constructor
	Stopwatch();
	Stopwatch(const Stopwatch& t_stopwatch) = default;
	Stopwatch(Stopwatch&& t_stopwatch) = default;

	// Destructor
	~Stopwatch() = default;

	// Operators
	Stopwatch& operator=(const Stopwatch& t_stopwatch) = default;
	Stopwatch& operator=(Stopwatch&& t_stopwatch) = default;

	// Functionality
	void start();
	void resume();
	void pause();
	void stop();

	// Getters
	float getTime() const;
	bool isStopped() const;
	bool isPaused() const;
	bool isRunning() const;
private:
	enum class State : unsigned char
	{
		Stopped,
		Paused,
		Running
	};

	// Attributes
	State m_state;
	double m_time;
	std::chrono::steady_clock::time_point m_start;
};

}
