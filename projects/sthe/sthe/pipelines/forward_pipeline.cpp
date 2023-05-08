#include "forward_pipeline.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/config/binding.hpp>
#include <sthe/util/io.hpp>
#include <sthe/core/application.hpp>
#include <sthe/core/scene.hpp>
#include <sthe/core/environment.hpp>
#include <sthe/core/mesh.hpp>
#include <sthe/core/sub_mesh.hpp>
#include <sthe/core/material.hpp>
#include <sthe/core/terrain.hpp>
#include <sthe/core/terrain_layer.hpp>
#include <sthe/components/transform.hpp>
#include <sthe/components/light.hpp>
#include <sthe/components/mesh_renderer.hpp>
#include <sthe/components/terrain_renderer.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <memory>
#include <array>
#include <vector>

namespace sthe
{

// Constructors
ForwardPipeline::ForwardPipeline() :
	m_meshProgram{ std::make_shared<gl::Program>() },
	m_terrainProgram{ std::make_shared<gl::Program>() },
	m_pipelineBuffer{ std::make_shared<gl::Buffer>(static_cast<int>(sizeof(uniform::ForwardPipeline)), 1) }
{
	m_meshProgram->attachShader(gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "mesh/phong.vert" });
    m_meshProgram->attachShader(gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "mesh/phong.frag" });
    m_meshProgram->link();

	m_terrainProgram->setPatchVertexCount(4);
	m_terrainProgram->attachShader(gl::Shader{ GL_VERTEX_SHADER, getShaderPath() + "terrain/phong.vert" });
	m_terrainProgram->attachShader(gl::Shader{ GL_TESS_CONTROL_SHADER, getShaderPath() + "terrain/phong.tesc" });
	m_terrainProgram->attachShader(gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "terrain/phong.tese" });
	m_terrainProgram->attachShader(gl::Shader{ GL_GEOMETRY_SHADER, getShaderPath() + "terrain/phong.geom" });
	m_terrainProgram->attachShader(gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "terrain/phong.frag" });
	m_terrainProgram->link();
}

// Functionality
void ForwardPipeline::use()
{
	GL_CHECK_ERROR(glEnable(GL_DEPTH_TEST));
	GL_CHECK_ERROR(glEnable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glEnable(GL_CULL_FACE));
}

void ForwardPipeline::disuse()
{
	GL_CHECK_ERROR(glDisable(GL_DEPTH_TEST));
	GL_CHECK_ERROR(glDisable(GL_SCISSOR_TEST));
	GL_CHECK_ERROR(glDisable(GL_CULL_FACE));
}

void ForwardPipeline::render(const Scene& t_scene, const Camera& t_camera)
{
	setup(t_scene, t_camera);

	if (t_camera.hasFramebuffer())
	{
		t_camera.getFramebuffer()->bind();
	}
	else
	{
		gl::DefaultFramebuffer::bind();
	}

	GL_CHECK_ERROR(glScissor(0, 0, m_data.resolution.x, m_data.resolution.y));
	GL_CHECK_ERROR(glViewport(0, 0, m_data.resolution.x, m_data.resolution.y));

	const glm::vec4& clearColor{ t_camera.getClearColor() };
	GL_CHECK_ERROR(glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a));
	GL_CHECK_ERROR(glClear(t_camera.getClearMask()));

	meshRendererPass(t_scene);
	terrainRendererPass(t_scene);
}

void ForwardPipeline::setup(const Scene& t_scene, const Camera& t_camera)
{
	const Application& application{ getApplication() };
	const Environment& environment{ t_scene.getEnvironment() };

	m_data.time = application.getTime();
	m_data.deltaTime = application.getDeltaTime();
	m_data.resolution = t_camera.getResolution();
	m_data.projectionMatrix = t_camera.getProjectionMatrix();
	m_data.viewMatrix = t_camera.getViewMatrix();
	m_data.inverseViewMatrix = t_camera.getInverseViewMatrix();
	m_data.viewProjectionMatrix = m_data.projectionMatrix * m_data.viewMatrix;
	m_data.environment.ambientColor = environment.getAmbientColor();
	m_data.environment.ambientIntensity = environment.getAmbientIntensity();
	m_data.environment.fogColor = environment.getFogColor();
	m_data.environment.fogDensity = environment.getFogDensity();
	m_data.environment.fogMode = static_cast<unsigned int>(environment.getFogMode());
	m_data.environment.fogStart = environment.getFogStart();
	m_data.environment.fogEnd = environment.getFogEnd();
	m_data.environment.lightCount = 0;

	const auto lights{ t_scene.getComponents<Transform, Light>() };

	for (const entt::entity entity : lights)
	{
		const auto [transform, light] { lights.get<const Transform, const Light>(entity) };

		uniform::Light& lightData{ m_data.environment.lights[m_data.environment.lightCount++] };
		lightData.position = transform.getPosition();
		lightData.type = static_cast<unsigned int>(light.getType());
		lightData.color = light.getColor();
		lightData.intensity = light.getIntensity();
		lightData.attenuation = light.getAttenuation();
		lightData.range = light.getRange();
		lightData.direction = transform.getForward();
		lightData.spotOuterCutOff = glm::cos(glm::radians(light.getSpotAngle()));
		lightData.spotInnerCutOff = glm::cos(glm::radians((1.0f - light.getSpotBlend()) * light.getSpotAngle()));

		if (m_data.environment.lightCount >= m_data.environment.lights.size())
		{
			break;
		}
	}

	m_pipelineBuffer->bind(GL_UNIFORM_BUFFER, STHE_UNIFORM_BUFFER_PIPELINE);
	m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data), static_cast<int>(offsetof(uniform::ForwardPipeline, environment.lights)) +
		                     m_data.environment.lightCount * static_cast<int>(sizeof(uniform::Light)));
}

void ForwardPipeline::meshRendererPass(const Scene& t_scene)
{
	const Material* activeMaterial{ nullptr };
	const gl::Program* activeProgram{ nullptr };
	
	const auto meshRenderers{ t_scene.getComponents<Transform, MeshRenderer>() };
	
	for (const entt::entity entity : meshRenderers)
	{
		const MeshRenderer& meshRenderer{ meshRenderers.get<const MeshRenderer>(entity) };

		if (meshRenderer.hasMesh() && meshRenderer.getMaterialCount() > 0)
		{
			const Transform& transform{ meshRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::ForwardPipeline, modelMatrix)), 4 * sizeof(glm::mat4));

			const Mesh& mesh{ *meshRenderer.getMesh() };
			mesh.bind();

			const std::vector<std::shared_ptr<Material>>& materials{ meshRenderer.getMaterials() };

			for (int i{ 0 }, j{ 0 }; i < mesh.getSubMeshCount(); ++i, j = std::min(j + 1, meshRenderer.getMaterialCount() - 1))
			{
				if (activeMaterial != materials[j].get())
				{
					activeMaterial = materials[j].get();
					activeMaterial->bind();

					const gl::Program* const program{ activeMaterial->hasProgram() ? activeMaterial->getProgram().get() :
																					 m_meshProgram.get() };

					if (activeProgram != program)
					{
						activeProgram = program;
						activeProgram->use();
					}
					
					uniform::Material& materialData{ m_data.material };
					materialData.diffuseColor = activeMaterial->getDiffuseColor();
					materialData.opacity = activeMaterial->getOpacity();
					materialData.specularColor = activeMaterial->getSpecularColor();
					materialData.specularIntensity = activeMaterial->getSpecularIntensity();
					materialData.shininess = activeMaterial->getShininess();
					materialData.hasDiffuseMap = activeMaterial->hasDiffuseMap();
					materialData.hasNormalMap = activeMaterial->hasNormalMap();
					m_pipelineBuffer->upload(reinterpret_cast<char*>(&materialData), static_cast<int>(offsetof(uniform::ForwardPipeline, material)), sizeof(uniform::Material));
				}

				const SubMesh& subMesh{ mesh.getSubMesh(i) };
				const long long int offset{ subMesh.getFirstIndex() * static_cast<long long int>(sizeof(int)) };

				GL_CHECK_ERROR(glDrawElementsInstancedBaseInstance(subMesh.getDrawMode(), subMesh.getIndexCount(), GL_UNSIGNED_INT, reinterpret_cast<void*>(offset),
							   meshRenderer.getInstanceCount(), static_cast<GLuint>(meshRenderer.getBaseInstance())));
			}
		}
	}
}

void ForwardPipeline::terrainRendererPass(const Scene& t_scene)
{
	const CustomMaterial* activeMaterial{ nullptr };
	const gl::Program* activeProgram{ nullptr };

	const auto terrainRenderers{ t_scene.getComponents<Transform, TerrainRenderer>() };

	for (const entt::entity entity : terrainRenderers)
	{
		const TerrainRenderer& terrainRenderer{ terrainRenderers.get<const TerrainRenderer>(entity) };

		if (terrainRenderer.hasTerrain() && terrainRenderer.hasMaterial())
		{
			const Transform& transform{ terrainRenderers.get<const Transform>(entity) };

			m_data.modelMatrix = transform.getModelMatrix();
			m_data.inverseModelMatrix = transform.getInverseModelMatrix();
			m_data.modelViewMatrix = m_data.viewMatrix * m_data.modelMatrix;
			m_data.inverseModelViewMatrix = m_data.inverseModelMatrix * m_data.inverseViewMatrix;
			m_pipelineBuffer->upload(reinterpret_cast<char*>(&m_data.modelMatrix), static_cast<int>(offsetof(uniform::ForwardPipeline, modelMatrix)), 4 * sizeof(glm::mat4));

			const CustomMaterial* const material{ terrainRenderer.getMaterial().get() };

			if (activeMaterial != material)
			{
				activeMaterial = material;
				activeMaterial->bind();
			}

			const gl::Program* const program{ activeMaterial->hasProgram() ? activeMaterial->getProgram().get() :
																		     m_terrainProgram.get() };

			if (activeProgram != program)
			{
				activeProgram = program;
				activeProgram->use();
			}

			const Terrain& terrain{ *terrainRenderer.getTerrain() };
			terrain.bind();

			uniform::Terrain& terrainData{ m_data.terrain };
			terrainData.size = terrain.getSize();
			terrainData.subDivision = terrain.getResolution() / terrain.getDetail();
			terrainData.detail = terrain.getDetail();
			terrainData.hasHeightMap = terrain.hasHeightMap();
			terrainData.hasAlphaMap = terrain.hasAlphaMap();
			terrainData.layerCount = 0;

			for (const auto& terrainLayer : terrain.getLayers())
			{
				uniform::TerrainLayer& terrainLayerData{ terrainData.layers[terrainData.layerCount++] };
				terrainLayerData.diffuseColor = terrainLayer->getDiffuseColor();
				terrainLayerData.specularIntensity = terrainLayer->getSpecularIntensity();
				terrainLayerData.specularColor = terrainLayer->getSpecularColor();
				terrainLayerData.shininess = terrainLayer->getShininess();
				terrainLayerData.hasDiffuseMap = terrainLayer->hasDiffuseMap();
			}

			m_pipelineBuffer->upload(reinterpret_cast<char*>(&terrainData), static_cast<int>(offsetof(uniform::ForwardPipeline, terrain)), sizeof(uniform::Terrain));
			
			GL_CHECK_ERROR(glDrawArraysInstanced(GL_PATCHES, 0, 4, terrainData.subDivision * terrainData.subDivision));
		}
	}
}

}
