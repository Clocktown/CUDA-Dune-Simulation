#pragma once

#include "scene.hpp"
#include "game_object.hpp"
#include "mesh.hpp"
#include "material.hpp"
#include <sthe/gl/program.hpp>
#include <sthe/gl/texture2d.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <memory>
#include <string>
#include <bitset>
#include <vector>
#include <unordered_map>

namespace sthe
{

class Importer
{
public:
	// Constructors
	explicit Importer(const std::string& t_file, const unsigned int t_flags = 0);
	Importer(const Importer& t_importer) = delete;
	Importer(Importer&& t_importer) = default;

	// Destructor
	~Importer() = default;

	// Operators
	Importer& operator=(const Importer& t_importer) = delete;
	Importer& operator=(Importer&& t_importer) = default;

	// Functionality
	const aiScene* readFile(const std::string& t_file, const unsigned int t_flags = 0);
	GameObject& importModel(Scene& t_scene, const std::shared_ptr<gl::Program>& t_program = nullptr, const bool t_hasMipmap = true);
	const std::vector<std::shared_ptr<Mesh>>& importMeshes();
	const std::vector<std::shared_ptr<Material>>& importMaterials(const std::shared_ptr<gl::Program>& t_program = nullptr, const bool t_hasMipmap = true);
	const std::vector<std::shared_ptr<gl::Texture2D>>& importTextures(const bool t_hasMipmap = true);

	// Getters
	const aiScene* getData() const;
	const std::vector<std::shared_ptr<Mesh>>& getMeshes() const;
	const std::vector<std::shared_ptr<Material>>& getMaterials() const;
	const std::vector<std::shared_ptr<gl::Texture2D>>& getTextures() const;
	int getMeshCount() const;
	int getMaterialCount() const;
	int getTextureCount() const;
private:
	// Functionality
	GameObject& importModel(Scene& t_scene, const aiNode& t_node);
	void importMesh(const aiNode& t_node);
	void importTexture(const std::string& t_file, const bool t_hasMipmap);

	// Attributes
	Assimp::Importer m_importer;
	const aiScene* m_data;

	std::string m_path;
	std::vector<std::shared_ptr<Mesh>> m_meshes;
	std::vector<std::shared_ptr<Material>> m_materials;
	std::vector<std::shared_ptr<gl::Texture2D>> m_textures;
	std::unordered_map<std::string, size_t> m_meshIndices;
	std::unordered_map<std::string, size_t> m_textureIndices;
	std::bitset<3> m_flags;
};

}
