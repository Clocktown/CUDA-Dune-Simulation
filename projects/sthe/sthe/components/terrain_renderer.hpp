#pragma once

#include <sthe/core/component.hpp>
#include <sthe/core/terrain.hpp>
#include <sthe/core/custom_material.hpp>
#include <memory>
#include <vector>

namespace sthe
{

class TerrainRenderer : public Component
{
public:
	// Constructors
	explicit TerrainRenderer(const std::shared_ptr<Terrain>& t_terrain = nullptr, const std::shared_ptr<CustomMaterial>& t_material = nullptr);
	TerrainRenderer(const TerrainRenderer& t_terrainRenderer) = delete;
	TerrainRenderer(TerrainRenderer&& t_terrainRenderer) = default;

	// Destructor
	~TerrainRenderer() = default;

	// Operators
	TerrainRenderer& operator=(const TerrainRenderer& t_terrainRenderer) = delete;
	TerrainRenderer& operator=(TerrainRenderer&& t_terrainRenderer) = default;

	// Setter
	void setTerrain(const std::shared_ptr<Terrain>& t_terrain);
	void setMaterial(const std::shared_ptr<CustomMaterial>& t_material);

	// Getters
	const std::shared_ptr<Terrain>& getTerrain() const;
	bool hasTerrain() const;
	const std::shared_ptr<CustomMaterial>& getMaterial() const;
	bool hasMaterial() const;
private:
	// Attribute 
	std::shared_ptr<Terrain> m_terrain;
	std::shared_ptr<CustomMaterial> m_material;
};

}
