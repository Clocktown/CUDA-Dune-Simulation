#pragma once

#include <cuda_runtime.h>

namespace sthe
{
namespace cu
{

class Stopwatch
{
public:
	// Constructor
	Stopwatch();
	Stopwatch(const Stopwatch& t_stopwatch) = delete;
	Stopwatch(Stopwatch&& t_stopwatch) noexcept;

	// Destructor
	~Stopwatch();

	// Operators
	Stopwatch& operator=(const Stopwatch& t_stopwatch) = delete;
	Stopwatch& operator=(Stopwatch&& t_stopwatch) noexcept;

	// Functionality
	void start(const cudaStream_t t_stream = nullptr);
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

	// Functionality
	void update() const;

	// Attributes
	State m_state;
	mutable float m_time;
	cudaStream_t m_stream;
	cudaEvent_t m_start;
	cudaEvent_t m_end;
	mutable bool m_hasChanged;
};

}
}
