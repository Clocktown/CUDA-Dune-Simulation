#pragma once

#include "renderer.hpp"
#include <sthe/core/mesh.hpp>
#include <sthe/core/material.hpp>
#include <memory>
#include <vector>

namespace sthe
{

class MeshRenderer : public Renderer
{
public:
	// Constructors
	explicit MeshRenderer(const std::shared_ptr<Mesh>& t_mesh = nullptr);
	explicit MeshRenderer(const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material);
	MeshRenderer(const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials);
	MeshRenderer(const MeshRenderer& t_meshRenderer) = delete;
	MeshRenderer(MeshRenderer&& t_meshRenderer) = default;

	// Destructor
	~MeshRenderer() = default;

	// Operators
	MeshRenderer& operator=(const MeshRenderer& t_meshRenderer) = delete;
	MeshRenderer& operator=(MeshRenderer&& t_meshRenderer) = default;

	// Functionality
	void addMaterial(const std::shared_ptr<Material>& t_material);
	void removeMaterial(const int t_index);
	void removeMaterials();

	// Setter
	void setMesh(const std::shared_ptr<Mesh>& t_mesh);
	void setMaterial(const int t_index, const std::shared_ptr<Material>& t_material);
	void setMaterials(const std::vector<std::shared_ptr<Material>>& t_materials);
	
	// Getters
	const std::shared_ptr<Mesh>& getMesh() const;
	bool hasMesh() const;
	const std::shared_ptr<Material>& getMaterial(const int t_index) const;
	const std::vector<std::shared_ptr<Material>>& getMaterials() const;
	int getMaterialCount() const;
	bool hasMaterials() const;
private:
	// Attribute 
	std::shared_ptr<Mesh> m_mesh;
	std::vector<std::shared_ptr<Material>> m_materials;
};

}
