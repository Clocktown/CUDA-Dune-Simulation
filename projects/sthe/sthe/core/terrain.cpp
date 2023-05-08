#include "terrain.hpp"
#include "terrain_layer.hpp"
#include <sthe/config/config.hpp>
#include <sthe/config/binding.hpp>
#include <sthe/core/custom_material.hpp>
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <sthe/gl/texture2d.hpp>
#include <memory>
#include <vector>

namespace sthe
{

// Constructors
Terrain::Terrain(const glm::vec3& t_size) :
	m_size{ t_size },
	m_resolution{ 512 },
	m_detail{ 32 },
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_heightMap{ nullptr },
	m_alphaMap{ nullptr }
{
	m_layers.reserve(4);
}

Terrain::Terrain(const glm::vec3& t_size, const std::shared_ptr<TerrainLayer>& t_layer) :
	m_size{ t_size },
	m_resolution{ 512 },
	m_detail{ 32 },
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_heightMap{ nullptr },
	m_alphaMap{ nullptr },
	m_layers{ t_layer }
{
	STHE_ASSERT(t_layer != nullptr, "Layer cannot be nullptr");

	m_layers.reserve(4);
}

Terrain::Terrain(const glm::vec3& t_size, const std::vector<std::shared_ptr<TerrainLayer>>& t_layers) :
    m_size { t_size },
    m_resolution{ 32 },
    m_detail{ 16 },
    m_heightMap{ nullptr },
    m_alphaMap{ nullptr },
    m_layers{ t_layers }
{
	STHE_ASSERT(std::find(t_layers.begin(), t_layers.end(), nullptr) == t_layers.end(), "No layer can be nullptr");

	m_layers.reserve(4);
}

// Functionality
void Terrain::bind() const
{
	m_vertexArray->bind();

	if (hasHeightMap())
	{
		m_heightMap->bind(STHE_TEXTURE_UNIT_TERRAIN_HEIGHT);
	}

	if (hasAlphaMap())
	{
		m_alphaMap->bind(STHE_TEXTURE_UNIT_TERRAIN_ALPHA);
	}

	for (int i{ 0 }; i < getLayerCount(); ++i)
	{
		if (m_layers[i]->hasDiffuseMap())
		{
			m_layers[i]->getDiffuseMap()->bind(STHE_TEXTURE_UNIT_TERRAIN_LAYER0_DIFFUSE + i);
		}
	}
}

void Terrain::addLayer(const std::shared_ptr<TerrainLayer>& t_layer)
{
	STHE_ASSERT(getLayerCount() < 4, "Layer count must be lower than or equal to 4");
	STHE_ASSERT(t_layer != nullptr, "Layer cannot be nullptr");

	m_layers.emplace_back(t_layer);
}

void Terrain::removeLayer(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < getLayerCount(), "Index must refer to an existing layer");

	m_layers.erase(m_layers.begin() + t_index);
}

void Terrain::removeLayers()
{
	m_layers.clear();
}

// Setters
void Terrain::setSize(const glm::vec3& t_size)
{
	m_size = t_size;
}

void Terrain::setResolution(const int t_resolution)
{
	STHE_ASSERT(t_resolution > 0, "Resolution must be greater than 0");

	m_resolution = t_resolution;
}

void Terrain::setDetail(const int t_detail)
{
	STHE_ASSERT(t_detail > 0, "Detail must be greater than 0");

	m_detail = t_detail;
}

void Terrain::setHeightMap(const std::shared_ptr<sthe::gl::Texture2D>& t_heightMap)
{
	m_heightMap = t_heightMap;
}

void Terrain::setAlphaMap(const std::shared_ptr<gl::Texture2D>& t_alphaMap)
{
	m_alphaMap = t_alphaMap;
}

void Terrain::setLayer(const int t_index, const std::shared_ptr<TerrainLayer>& t_layer)
{
	STHE_ASSERT(t_index >= 0 && t_index < getLayerCount(), "Index must refer to an existing layer");
	STHE_ASSERT(t_layer != nullptr, "Layer cannot be nullptr");

	m_layers[t_index] = t_layer;
}

void Terrain::setLayers(const std::vector<std::shared_ptr<TerrainLayer>>& t_layers)
{
	STHE_ASSERT(t_layers.size() <= 4, "Index must refer to an existing layer");
	STHE_ASSERT(std::find(t_layers.begin(), t_layers.end(), nullptr) == t_layers.end(), "No layer can be nullptr");

	m_layers = t_layers;
}

// Getters
const glm::vec3& Terrain::getSize() const
{
	return m_size;
}

int Terrain::getResolution() const
{
	return m_resolution;
}

int Terrain::getDetail() const
{
	return m_detail;
}

const std::shared_ptr<gl::VertexArray> Terrain::getVertexArray() const
{
	return m_vertexArray;
}

const std::shared_ptr<gl::Texture2D>& Terrain::getHeightMap() const
{
	return m_heightMap;
}

const std::shared_ptr<gl::Texture2D>& Terrain::getAlphaMap() const
{
	return m_alphaMap;
}

const std::shared_ptr<TerrainLayer>& Terrain::getLayer(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < getLayerCount(), "Index must refer to an existing layer");

	return m_layers[t_index];
}

const std::vector<std::shared_ptr<TerrainLayer>>& Terrain::getLayers() const
{
	return m_layers;
}

int Terrain::getLayerCount() const
{
	return static_cast<int>(m_layers.size());
}

bool Terrain::hasHeightMap() const
{
	return m_heightMap != nullptr;
}

bool Terrain::hasAlphaMap() const
{
	return m_alphaMap != nullptr;
}

bool Terrain::hasLayers() const
{
	return !m_layers.empty();
}

}
