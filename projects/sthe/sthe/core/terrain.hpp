#pragma once

#include "terrain_layer.hpp"
#include <sthe/core/custom_material.hpp>
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture2d.hpp>
#include <memory>
#include <array>
#include <vector>

namespace sthe
{

namespace uniform
{

struct Terrain
{
	glm::vec3 size;
	int subDivision;
	int detail;
	int layerCount;
	int hasHeightMap;
	int hasAlphaMap;
	std::array<TerrainLayer, 4> layers;
};

}

class Terrain
{
public:
	// Constructors
	Terrain(const glm::vec3& t_size = glm::vec3{ 512.0f, 16.0f, 512.0f });
	Terrain(const glm::vec3& t_size, const std::shared_ptr<TerrainLayer>& t_layer);
	Terrain(const glm::vec3& t_size, const std::vector<std::shared_ptr<TerrainLayer>>& t_layers);
	Terrain(const Terrain& t_terrain) = default;
	Terrain(Terrain&& t_terrain) = default;

	// Destructor
	~Terrain() = default;

	// Operators
	Terrain& operator=(const Terrain& t_terrain) = default;
	Terrain& operator=(Terrain&& t_terrain) = default;

	// Functionality
	void bind() const;
	void addLayer(const std::shared_ptr<TerrainLayer>& t_layer);
	void removeLayer(const int t_index);
	void removeLayers();

	// Setters
	void setSize(const glm::vec3& t_size);
	void setResolution(const int t_resolution);
	void setDetail(const int t_detail);
	void setHeightMap(const std::shared_ptr<gl::Texture2D>& t_heightMap);
	void setAlphaMap(const std::shared_ptr<gl::Texture2D>& t_alphaMap);
	void setLayer(const int t_index, const std::shared_ptr<TerrainLayer>& t_layer);
	void setLayers(const std::vector<std::shared_ptr<TerrainLayer>>& t_layers);
	
	// Getters
	const glm::vec3& getSize() const;
	int getResolution() const;
	int getDetail() const;
	const std::shared_ptr<gl::VertexArray> getVertexArray() const;
	const std::shared_ptr<gl::Texture2D>& getHeightMap() const;
	const std::shared_ptr<gl::Texture2D>& getAlphaMap() const;
	const std::shared_ptr<TerrainLayer>& getLayer(const int t_index) const;
	const std::vector<std::shared_ptr<TerrainLayer>>& getLayers() const;
	int getLayerCount() const;
	bool hasHeightMap() const;
	bool hasAlphaMap() const;
	bool hasLayers() const;
private:
	// Attributes
	glm::vec3 m_size;
	int m_resolution;
	int m_detail;
	std::shared_ptr<gl::VertexArray> m_vertexArray;
	std::shared_ptr<gl::Texture2D> m_heightMap;
	std::shared_ptr<gl::Texture2D> m_alphaMap;
	std::vector<std::shared_ptr<TerrainLayer>> m_layers;
};

}
