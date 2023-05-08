#include "stopwatch.hpp"
#include <sthe/config/debug.hpp>
#include <chrono>

namespace sthe
{

// Constructor
Stopwatch::Stopwatch() :
	m_state{ State::Stopped },
	m_time{ 0.0f }
{

}

// Functionality
void Stopwatch::start()
{
	STHE_ASSERT(m_state == State::Stopped, "Stopwatch must be stopped");

	m_state = State::Running;
	m_time = 0.0f;
	m_start = std::chrono::steady_clock::now();
}

void Stopwatch::resume()
{
	STHE_ASSERT(m_state == State::Paused, "Stopwatch must be paused");

	m_state = State::Running;
	m_start = std::chrono::steady_clock::now();
}

void Stopwatch::pause()
{
	STHE_ASSERT(m_state == State::Running, "Stopwatch must be running");

	const std::chrono::duration<double, std::micro> duration{ std::chrono::steady_clock::now() - m_start };

	m_state = State::Paused;
	m_time += duration.count() / 1000.0;
}

void Stopwatch::stop()
{
	STHE_ASSERT(m_state != State::Stopped, "Stopwatch must not be stopped");

	if (m_state == State::Running)
	{
		const std::chrono::duration<double, std::micro> duration{ std::chrono::steady_clock::now() - m_start };
		m_time += duration.count() / 1000.0;
	}

	m_state = State::Stopped;
}

// Getters
float Stopwatch::getTime() const
{
	if (m_state == State::Running)
	{
		const std::chrono::duration<double, std::micro> duration{ std::chrono::steady_clock::now() - m_start };
		return static_cast<float>(m_time + duration.count() / 1000.0);
	}

	return static_cast<float>(m_time);
}

bool Stopwatch::isStopped() const
{
	return m_state == State::Stopped;
}

bool Stopwatch::isPaused() const
{
	return m_state == State::Paused;
}

bool Stopwatch::isRunning() const
{
	return m_state == State::Running;
}

}
