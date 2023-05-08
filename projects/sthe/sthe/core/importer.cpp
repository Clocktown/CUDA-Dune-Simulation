#include "importer.hpp"
#include "scene.hpp"
#include "game_object.hpp"
#include "mesh.hpp"
#include "sub_mesh.hpp"
#include "material.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <sthe/components/mesh_renderer.hpp>
#include <sthe/gl/program.hpp>
#include <sthe/gl/texture2d.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <stb_image.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#define STHE_IMPORTER_MESH_BIT 0 
#define STHE_IMPORTER_MATERIAL_BIT 1
#define STHE_IMPORTER_TEXTURE_BIT 2

namespace sthe
{

// Constructor
Importer::Importer(const std::string& t_file, const unsigned int t_flags) :
	m_data{ m_importer.ReadFile(t_file, t_flags) }
{
	STHE_ASSERT(m_data != nullptr, "Failed to read file");

	const std::filesystem::path path{ t_file };

	if (path.has_parent_path())
	{
		m_path = path.parent_path().string() + std::string{ std::filesystem::path::preferred_separator };
	}
}

// Functionality
const aiScene* Importer::readFile(const std::string& t_file, const unsigned int t_flags)
{
	m_data = m_importer.ReadFile(t_file, t_flags);

	STHE_ASSERT(m_data != nullptr, "Failed to read file");

	const std::filesystem::path path{ t_file };

	if (path.has_parent_path())
	{
		m_path = path.parent_path().string() + std::string{ std::filesystem::path::preferred_separator };
	}
	else
	{
		m_path.clear();
	}

	m_meshes.clear();
	m_materials.clear();
	m_textures.clear();
	m_flags.reset();

	return m_data;
}

GameObject& Importer::importModel(Scene& t_scene, const std::shared_ptr<gl::Program>& t_program, const bool t_hasMipmap)
{
	importMeshes();
	importMaterials(t_program, t_hasMipmap);

	return importModel(t_scene, *m_data->mRootNode);
}

const std::vector<std::shared_ptr<Mesh>>& Importer::importMeshes()
{
	if (m_flags.test(STHE_IMPORTER_MESH_BIT))
	{
		return m_meshes;
	}

	importMesh(*m_data->mRootNode);

	m_flags.set(STHE_IMPORTER_MESH_BIT);

	return m_meshes;
}

const std::vector<std::shared_ptr<Material>>& Importer::importMaterials(const std::shared_ptr<gl::Program>& t_program, const bool t_hasMipmap)
{
	if (m_flags.test(STHE_IMPORTER_MATERIAL_BIT))
	{
		return m_materials;
	}

	m_materials.reserve(m_data->mNumMaterials);
	importTextures(t_hasMipmap);

	for (unsigned int i{ 0 }; i < m_data->mNumMaterials; ++i)
	{
		const aiMaterial& materialData{ *m_data->mMaterials[i] };
		aiColor3D color;
		float value;
		aiString file;
	
		Material& material{ *m_materials.emplace_back(std::make_shared<Material>(t_program)) };

		if (materialData.Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
		{
			material.setDiffuseColor(glm::vec3{ color.r, color.g, color.b });
		}

		if (materialData.Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
		{
			material.setSpecularColor(glm::vec3{ color.r, color.g, color.b });
		}

		if (materialData.Get(AI_MATKEY_SHININESS_STRENGTH, value) == AI_SUCCESS)
		{
			material.setSpecularIntensity(value);
		}

		if (materialData.Get(AI_MATKEY_SHININESS, value) == AI_SUCCESS)
		{
			material.setShininess(value);
		}

		if (materialData.Get(AI_MATKEY_OPACITY, value) == AI_SUCCESS)
		{
			material.setOpacity(value);
		}

		if (materialData.Get(AI_MATKEY_TEXTURE_DIFFUSE(0), file) == AI_SUCCESS)
		{
			material.setDiffuseMap(m_textures[m_textureIndices[std::string{ file.C_Str() }]]);
		}

		if (materialData.Get(AI_MATKEY_TEXTURE_NORMALS(0), file) == AI_SUCCESS)
		{
			material.setNormalMap(m_textures[m_textureIndices[std::string{ file.C_Str() }]]);
		}
	}

	m_flags.set(STHE_IMPORTER_MATERIAL_BIT);

	return m_materials;
}

const std::vector<std::shared_ptr<gl::Texture2D>>& Importer::importTextures(const bool t_hasMipmap)
{
	if (m_flags.test(STHE_IMPORTER_TEXTURE_BIT))
	{
		return m_textures;
	}

	for (unsigned int i{ 0 }; i < m_data->mNumTextures; ++i)
	{
		const aiTexture& textureData{ *m_data->mTextures[i] };
		std::shared_ptr<gl::Texture2D> texture2D;

		if (textureData.mHeight > 0)
		{
			texture2D = std::make_shared<gl::Texture2D>(static_cast<int>(textureData.mWidth), static_cast<int>(textureData.mHeight), GL_RGBA8, t_hasMipmap);
			texture2D->upload(textureData.pcData, texture2D->getWidth(), texture2D->getHeight(), GL_BGRA, GL_UNSIGNED_BYTE);
			texture2D->recalculateMipmap();
		}
		else
		{
			int width;
			int height;
			GLenum internalFormat;
			GLenum pixelFormat;
			int channels;
			stbi_set_flip_vertically_on_load(1);
			stbi_uc* const source{ stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(textureData.pcData), textureData.mWidth, &width, &height, &channels, 0) };

			STHE_ASSERT(source != nullptr, "Failed to decode data");

			switch (channels)
			{
			case 1:
				internalFormat = GL_R8;
				pixelFormat = GL_RED;
				break;
			case 2:
				internalFormat = GL_RG8;
				pixelFormat = GL_RG;
				break;
			case 3:
				internalFormat = GL_RGB8;
				pixelFormat = GL_RGB;
				break;
			case 4:
				internalFormat = GL_RGBA8;
				pixelFormat = GL_RGBA;
				break;
			default:
				STHE_ERROR("Channels must be between 1 and 4");
				break;
			}

			texture2D = std::make_shared<gl::Texture2D>(width, height, internalFormat, t_hasMipmap);
			texture2D->upload(source, width, height, pixelFormat, GL_UNSIGNED_BYTE);
			texture2D->recalculateMipmap();

			stbi_image_free(source);
		}

		m_textureIndices.emplace("*" + std::to_string(i), m_textures.size());
		m_textures.emplace_back(texture2D);
	}

	for (unsigned int i{ 0 }; i < m_data->mNumMaterials; ++i)
	{
		const aiMaterial& materialData{ *m_data->mMaterials[i] };
		aiString file;

		if (materialData.Get(AI_MATKEY_TEXTURE_DIFFUSE(0), file) == AI_SUCCESS)
		{
			importTexture(std::string{ file.C_Str() }, t_hasMipmap);
		}

		if (materialData.Get(AI_MATKEY_TEXTURE_NORMALS(0), file) == AI_SUCCESS)
		{
			importTexture(std::string{ file.C_Str() }, t_hasMipmap);
		}
	}

	m_flags.set(STHE_IMPORTER_TEXTURE_BIT);

	return m_textures;
}

GameObject& Importer::importModel(Scene& t_scene, const aiNode& t_node)
{
	GameObject& gameObject{ t_scene.addGameObject(std::string{ t_node.mName.C_Str() }) };
	Transform& transform{ gameObject.getTransform() };

	aiVector3D position;
	aiQuaternion rotation;
	aiVector3D scale;
	t_node.mTransformation.Decompose(scale, rotation, position);
	transform.setLocalPositionAndRotation(glm::vec3{ position.x, position.y, position.z }, glm::quat{ rotation.w, rotation.x, rotation.y, rotation.z });
	transform.setLocalScale(glm::vec3{ scale.x, scale.y, scale.z });

	if (t_node.mNumMeshes > 0)
	{
		MeshRenderer& meshRenderer{ gameObject.addComponent<MeshRenderer>() };
		std::string key;
		
		for (unsigned int i{ 0 }; i < t_node.mNumMeshes; ++i)
		{
			const aiMesh& meshData{ *m_data->mMeshes[t_node.mMeshes[i]] };

			meshRenderer.addMaterial(m_materials[meshData.mMaterialIndex]);
			key += "*" + std::to_string(t_node.mMeshes[i]);
		}

		meshRenderer.setMesh(m_meshes[m_meshIndices[key]]);
	}

	for (unsigned int i{ 0 }; i < t_node.mNumChildren; ++i)
	{
		GameObject& child{ importModel(t_scene, *t_node.mChildren[i]) };
		child.getTransform().setParent(&transform, false);
	}

	return gameObject;
}

void Importer::importMesh(const aiNode& t_node)
{
	if (t_node.mNumMeshes > 0)
	{
		std::string key;
		unsigned int indexCount{ 0 };
		unsigned int vertexCount{ 0 };
		bool hasNormals{ false };
		bool hasTangents{ false };
		bool hasUVs{ false };

		for (unsigned int i{ 0 }; i < t_node.mNumMeshes; ++i)
		{
			const aiMesh& meshData{ *m_data->mMeshes[t_node.mMeshes[i]] };

			STHE_ASSERT(meshData.HasFaces() && meshData.HasPositions(), "Mesh must have faces and positions");
		    STHE_ASSERT(meshData.mPrimitiveTypes == static_cast<unsigned int>(aiPrimitiveType_TRIANGLE), "Primitive must be a triangle");

			key += "*" + std::to_string(t_node.mMeshes[i]);
			indexCount += 3 * meshData.mNumFaces;
			vertexCount += meshData.mNumVertices;
			hasNormals |= meshData.HasNormals();
			hasTangents |= meshData.HasTangentsAndBitangents();
			hasUVs |= meshData.HasTextureCoords(0);
		}

		if (!m_meshIndices.contains(key))
		{
			std::vector<SubMesh> subMeshes;
			std::vector<int> indices;
			std::vector<glm::vec4> positions;
			std::vector<glm::vec4> normals;
			std::vector<glm::vec4> tangents;
			std::vector<glm::vec2> uvs;

			subMeshes.reserve(t_node.mNumMeshes);
			indices.reserve(indexCount);
			positions.reserve(vertexCount);

			if (hasNormals)
			{
				normals.reserve(vertexCount);
			}

			if (hasTangents)
			{
				tangents.reserve(vertexCount);
			}

			if (hasUVs)
			{
				uvs.reserve(vertexCount);
			}

			vertexCount = 0;

			for (unsigned int i{ 0 }; i < t_node.mNumMeshes; ++i)
			{
				const aiMesh& subMeshData{ *m_data->mMeshes[t_node.mMeshes[i]] };

				subMeshes.emplace_back(static_cast<int>(indices.size()), 3 * static_cast<int>(subMeshData.mNumFaces));

				for (unsigned int j{ 0 }; j < subMeshData.mNumFaces; ++j)
				{
					const aiFace& face{ subMeshData.mFaces[j] };
					indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[0]));
					indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[1]));
					indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[2]));
				}

				for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
				{
					const aiVector3D& position{ subMeshData.mVertices[j] };
					positions.emplace_back(position.x, position.y, position.z, 1.0f);
				}

				vertexCount += subMeshData.mNumVertices;

				if (subMeshData.HasNormals())
				{
					for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
					{
						const aiVector3D& normal{ subMeshData.mNormals[j] };
						normals.emplace_back(normal.x, normal.y, normal.z, 0.0f);
					}
				}
				else
				{
					normals.resize(vertexCount, glm::vec4{ 0.0f });
				}

				if (subMeshData.HasTangentsAndBitangents())
				{
					for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
					{
						const aiVector3D& tangent{ subMeshData.mTangents[j] };
						tangents.emplace_back(tangent.x, tangent.y, tangent.z, 0.0f);
					}
				}
				else
				{
					tangents.resize(vertexCount, glm::vec4{ 0.0f });
				}

				if (subMeshData.HasTextureCoords(0))
				{
					for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
					{
						const aiVector3D& uv{ subMeshData.mTextureCoords[0][j] };
						uvs.emplace_back(uv.x, uv.y);
					}
				}
				else
				{
					uvs.resize(vertexCount, glm::vec2{ 0.0f });
				}
			}

			const std::shared_ptr<Mesh> mesh{ std::make_shared<Mesh>() };
			mesh->setSubMeshes(subMeshes);
			mesh->setIndices(indices);
			mesh->setPositions(positions);
			mesh->setNormals(normals);
			mesh->setTangents(tangents);
			mesh->setUVs(uvs);
			mesh->upload();

			m_meshIndices.emplace(key, m_meshes.size());
			m_meshes.emplace_back(mesh);
		}
	}

	for (unsigned int i{ 0 }; i < t_node.mNumChildren; ++i)
	{
		importMesh(*t_node.mChildren[i]);
	}
}

void Importer::importTexture(const std::string& t_file, const bool t_hasMipmap)
{
	if (!m_textureIndices.contains(t_file))
	{
		m_textureIndices.emplace(t_file, m_textures.size());
		m_textures.emplace_back(std::make_shared<gl::Texture2D>(m_path + t_file, t_hasMipmap));
	}
}

// Getters
const aiScene* Importer::getData() const
{
	return m_data;
}

const std::vector<std::shared_ptr<Mesh>>& Importer::getMeshes() const
{
	return m_meshes;
}

const std::vector<std::shared_ptr<Material>>& Importer::getMaterials() const
{
	return m_materials;
}

const std::vector<std::shared_ptr<gl::Texture2D>>& Importer::getTextures() const
{
	return m_textures;
}

int Importer::getMeshCount() const
{
	return static_cast<int>(m_meshes.size());
}

int Importer::getMaterialCount() const
{
	return static_cast<int>(m_materials.size());
}

int Importer::getTextureCount() const
{
	return static_cast<int>(m_textures.size());
}

}
