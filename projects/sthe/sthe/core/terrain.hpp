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
	glm::ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
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
	explicit Terrain(const glm::ivec2& t_gridSize = glm::ivec2{ 512 }, const float t_gridScale = 1.0f, const float t_heightScale = 1.0f);
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
	void setGridSize(const glm::ivec2& t_gridSize);
	void setGridScale(const float t_gridScale);
	void setHeightScale(const float t_heightScale);
	void setTesselationLevel(const int t_tesselationLevel);
	void setHeightMap(const std::shared_ptr<gl::Texture2D>& t_heightMap);
	void setAlphaMap(const std::shared_ptr<gl::Texture2D>& t_alphaMap);
	void setLayer(const int t_index, const std::shared_ptr<TerrainLayer>& t_layer);
	void setLayers(const std::vector<std::shared_ptr<TerrainLayer>>& t_layers);
	
	// Getters
	const glm::ivec2& getGridSize() const;
	const float getGridScale() const;
	const float getHeightScale() const;
	int getTesselationLevel() const;
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
	glm::ivec2 m_gridSize;
	float m_gridScale;
	float m_heightScale;
	int m_tesselationLevel;
	std::shared_ptr<gl::VertexArray> m_vertexArray;
	std::shared_ptr<gl::Texture2D> m_heightMap;
	std::shared_ptr<gl::Texture2D> m_alphaMap;
	std::vector<std::shared_ptr<TerrainLayer>> m_layers;
};

}
