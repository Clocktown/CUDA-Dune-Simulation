#include "mesh_renderer.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/core/mesh.hpp>
#include <sthe/core/material.hpp>
#include <memory>
#include <vector>

namespace sthe
{

// Constructors
MeshRenderer::MeshRenderer(const std::shared_ptr<Mesh>& t_mesh) :
	m_mesh{ t_mesh }
{
}

MeshRenderer::MeshRenderer(const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material) :
	m_mesh{ t_mesh },
	m_materials{ t_material }
{
	STHE_ASSERT(t_material != nullptr, "Material cannot be nullptr");
}

MeshRenderer::MeshRenderer(const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials) :
	m_mesh{ t_mesh },
	m_materials{ t_materials }
{
	STHE_ASSERT(std::find(t_materials.begin(), t_materials.end(), nullptr) == t_materials.end(), "No material can be nullptr");
}

// Functionality
void MeshRenderer::addMaterial(const std::shared_ptr<Material>& t_material)
{
	STHE_ASSERT(t_material != nullptr, "Material cannot be nullptr");

	m_materials.emplace_back(t_material);
}

void MeshRenderer::removeMaterial(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < getMaterialCount(), "Index must refer to an existing material");

	m_materials.erase(m_materials.begin() + t_index);
}

void MeshRenderer::removeMaterials()
{
	m_materials.clear();
}

// Setters
void MeshRenderer::setMesh(const std::shared_ptr<Mesh>& t_mesh)
{
	m_mesh = t_mesh;
}

void MeshRenderer::setMaterial(const int t_index, const std::shared_ptr<Material>& t_material)
{
	STHE_ASSERT(t_index >= 0 && t_index < getMaterialCount(), "Index must refer to an existing material");
	STHE_ASSERT(t_material != nullptr, "Material cannot be nullptr");

	m_materials[t_index] = t_material;
}

void MeshRenderer::setMaterials(const std::vector<std::shared_ptr<Material>>& t_materials)
{
	STHE_ASSERT(std::find(t_materials.begin(), t_materials.end(), nullptr) == t_materials.end(), "No material can be nullptr");

	m_materials = t_materials;
}

// Getters
const std::shared_ptr<Mesh>& MeshRenderer::getMesh() const
{
	return m_mesh;
}

const std::shared_ptr<Material>& MeshRenderer::getMaterial(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < getMaterialCount(), "Index must refer to an existing material");

	return m_materials[t_index];
}

const std::vector<std::shared_ptr<Material>>& MeshRenderer::getMaterials() const
{
	return m_materials;
}

int MeshRenderer::getMaterialCount() const
{
	return static_cast<int>(m_materials.size());
}

bool MeshRenderer::hasMesh() const
{
	return m_mesh != nullptr;
}

bool MeshRenderer::hasMaterials() const
{
	return !m_materials.empty();
}

}
