#include "terrain_renderer.hpp"
#include <sthe/core/terrain.hpp>
#include <sthe/core/custom_material.hpp>
#include <memory>
#include <vector>

namespace sthe
{

// Constructors
TerrainRenderer::TerrainRenderer(const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material) :
	m_terrain{ t_terrain },
	m_material{ t_material }
{

}

// Setters
void TerrainRenderer::setTerrain(const std::shared_ptr<Terrain>& t_terrain)
{
	m_terrain = t_terrain;
}

void TerrainRenderer::setMaterial(const std::shared_ptr<CustomMaterial>& t_material)
{
	m_material = t_material;
}

// Getters
const std::shared_ptr<Terrain>& TerrainRenderer::getTerrain() const
{
	return m_terrain;
}

const std::shared_ptr<CustomMaterial>& TerrainRenderer::getMaterial() const
{
	return m_material;
}

bool TerrainRenderer::hasTerrain() const
{
	return m_terrain != nullptr;
}

bool TerrainRenderer::hasMaterial() const
{
	return m_material != nullptr;
}

}
