#include "stopwatch.hpp"
#include <sthe/config/debug.hpp>
#include <cuda_runtime.h>
#include <utility>

namespace sthe
{
namespace cu 
{

// Constructors
Stopwatch::Stopwatch() :
	m_state{ State::Stopped },
	m_time{ 0.0f },
	m_stream{ nullptr },
	m_hasChanged{ false }
{
	CU_CHECK_ERROR(cudaEventCreate(&m_start));
	CU_CHECK_ERROR(cudaEventCreate(&m_end));
}

Stopwatch::Stopwatch(Stopwatch&& t_stopwatch) noexcept :
	m_state{ std::exchange(t_stopwatch.m_state, State::Stopped) },
	m_time{ std::exchange(t_stopwatch.m_time, 0.0f) },
	m_stream{ std::exchange(t_stopwatch.m_stream, nullptr) },
	m_start{ std::exchange(t_stopwatch.m_start, nullptr) },
	m_end{ std::exchange(t_stopwatch.m_end, nullptr) },
	m_hasChanged{ std::exchange(t_stopwatch.m_hasChanged, false) }
{

}

// Destructor
Stopwatch::~Stopwatch()
{
	CU_CHECK_ERROR(cudaEventDestroy(m_start));
	CU_CHECK_ERROR(cudaEventDestroy(m_end));
}

// Operator
Stopwatch& Stopwatch::operator=(Stopwatch&& t_stopwatch) noexcept
{
	if (this != &t_stopwatch)
	{
		CU_CHECK_ERROR(cudaEventDestroy(m_start));
		CU_CHECK_ERROR(cudaEventDestroy(m_end));

		m_state = std::exchange(t_stopwatch.m_state, State::Stopped);
		m_time = std::exchange(t_stopwatch.m_time, 0.0f);
		m_stream = std::exchange(t_stopwatch.m_stream, nullptr);
		m_start = std::exchange(t_stopwatch.m_start, nullptr);
		m_end = std::exchange(t_stopwatch.m_end, nullptr);
		m_hasChanged = std::exchange(t_stopwatch.m_hasChanged, false);
	}

	return *this;
}

// Functionality
void Stopwatch::start(const cudaStream_t t_stream)
{
	STHE_ASSERT(m_state == State::Stopped, "Stopwatch must be stopped");

	m_state = State::Running;
	m_time = 0.0f;
	m_stream = t_stream;
	m_hasChanged = false;
	CU_CHECK_ERROR(cudaEventRecord(m_start, m_stream));
}

void Stopwatch::resume()
{
	STHE_ASSERT(m_state == State::Paused, "Stopwatch must be paused");

	update();

	m_state = State::Running;
	CU_CHECK_ERROR(cudaEventRecord(m_start, m_stream));
}

void Stopwatch::pause()
{
	STHE_ASSERT(m_state == State::Running, "Stopwatch must be running");

	CU_CHECK_ERROR(cudaEventRecord(m_end, m_stream));
	m_state = State::Paused;
	m_hasChanged = true;
}

void Stopwatch::stop()
{
	STHE_ASSERT(m_state != State::Stopped, "Stopwatch must be running or be paused");

	if (m_state == State::Running)
	{
		CU_CHECK_ERROR(cudaEventRecord(m_end, m_stream));
		m_hasChanged = true;
	}

	m_state = State::Stopped;
}

void Stopwatch::update() const
{
	if (m_hasChanged)
	{
		float duration;
		CU_CHECK_ERROR(cudaEventSynchronize(m_end));
		CU_CHECK_ERROR(cudaEventElapsedTime(&duration, m_start, m_end));

		m_time += duration;
		m_hasChanged = false;
	}
}

// Getters
float Stopwatch::getTime() const
{
	if (m_state == State::Running)
	{
		float duration;
		CU_CHECK_ERROR(cudaEventSynchronize(m_end));
		CU_CHECK_ERROR(cudaEventElapsedTime(&duration, m_start, m_end));

		return m_time + duration;
	}

	update();

	return m_time;
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
}
