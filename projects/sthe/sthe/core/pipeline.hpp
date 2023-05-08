#pragma once

#include "scene.hpp"
#include <sthe/components/camera.hpp>
#include <sthe/gl/program.hpp>
#include <memory>

namespace sthe
{

class Pipeline
{
public:
	// Constructor
	Pipeline() = default;
	Pipeline(const Pipeline& t_pipeline) = delete;
	Pipeline(Pipeline&& t_pipeline) = delete;

	// Destructor
	virtual ~Pipeline() = default;

	// Operators
	Pipeline& operator=(const Pipeline& t_pipeline) = delete;
	Pipeline& operator=(Pipeline&& t_pipeline) = delete;

	// Functionality
	virtual void use();
	virtual void disuse();
	void render(const Scene& t_scene);
	virtual void render(const Scene& t_scene, const Camera& t_camera) = 0;
};

}
