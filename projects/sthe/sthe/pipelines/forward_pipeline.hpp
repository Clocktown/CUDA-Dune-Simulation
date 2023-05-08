#pragma once

#include <sthe/core/pipeline.hpp>
#include <sthe/core/scene.hpp>
#include <sthe/core/terrain.hpp>
#include <sthe/core/material.hpp>
#include <sthe/components/camera.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <memory>
#include <array>
#include <vector>

namespace sthe
{

namespace uniform
{

struct ForwardPipeline
{
	float time;
	float deltaTime;
	glm::ivec2 resolution;
	glm::mat4 projectionMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 inverseViewMatrix;
	glm::mat4 viewProjectionMatrix;
	Environment environment;

	glm::mat4 modelMatrix;
	glm::mat4 inverseModelMatrix;
	glm::mat4 modelViewMatrix;
	glm::mat4 inverseModelViewMatrix;

	union
	{
		Material material;
		Terrain terrain;
	};
};

}

class ForwardPipeline : public Pipeline
{
public:
	// Constructors
	ForwardPipeline();
	ForwardPipeline(const ForwardPipeline& t_forwardPipeline) = delete;
	ForwardPipeline(ForwardPipeline&& t_forwardPipeline) = default;

	// Destructor
	~ForwardPipeline() = default;

	// Operators
	ForwardPipeline& operator=(const ForwardPipeline& t_forwardPipeline) = delete;
	ForwardPipeline& operator=(ForwardPipeline&& t_forwardPipeline) = default;

	// Functionality
	void use() override;
	void disuse() override;
	void render(const Scene& t_scene, const Camera& t_camera) override;
private:
	// Functionality
	void setup(const Scene& t_scene, const Camera& t_camera);
	void meshRendererPass(const Scene& t_scene);
	void terrainRendererPass(const Scene& t_scene);

	// Attributes
	uniform::ForwardPipeline m_data;
	std::shared_ptr<gl::Program> m_meshProgram;
	std::shared_ptr<gl::Program> m_terrainProgram;
	std::shared_ptr<gl::Buffer> m_pipelineBuffer;
};

}
