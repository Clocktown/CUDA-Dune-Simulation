#include "mesh.hpp"
#include "sub_mesh.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/config/binding.hpp>
#include <sthe/util/io.hpp>
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <memory>
#include <array>
#include <vector>
#include <bitset>

#define STHE_MESH_INDEX_BIT 0
#define STHE_MESH_POSITION_BIT 1
#define STHE_MESH_NORMAL_BIT 2
#define STHE_MESH_TANGENT_BIT 3
#define STHE_MESH_UV_BIT 4

namespace sthe
{
	
void ScreenFillingQuad::draw()
{
	static const Mesh quad{ Geometry::Quad };

	quad.bind();
	GL_CHECK_ERROR(glDrawElements(GL_TRIANGLES, quad.getSubMesh(0).getIndexCount(), GL_UNSIGNED_INT, nullptr));
}

static const std::array<std::string, 6> geometryFiles{ getMeshPath() + "quad.obj", getMeshPath() + "plane.obj", getMeshPath() + "cube.obj",
													   getMeshPath() + "cylinder.obj", getMeshPath() + "cone.obj", getMeshPath() + "sphere.obj" };

// Constructors
Mesh::Mesh() :
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_indexBuffer{ std::make_shared<gl::Buffer>() },
	m_positionBuffer{ std::make_shared<gl::Buffer>() },
	m_normalBuffer{ std::make_shared<gl::Buffer>() },
	m_tangentBuffer{ std::make_shared<gl::Buffer>() },
	m_uvBuffer{ std::make_shared<gl::Buffer>() }
{
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_POSITION, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_NORMAL, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_TANGENT, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_UV, 2, GL_FLOAT);
}

Mesh::Mesh(const Geometry t_geometry) :
	Mesh{ geometryFiles[static_cast<size_t>(t_geometry)] }
{
	
}

Mesh::Mesh(const std::string& t_file, const unsigned int t_flags) :
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_indexBuffer{ std::make_shared<gl::Buffer>() },
	m_positionBuffer{ std::make_shared<gl::Buffer>() },
	m_normalBuffer{ std::make_shared<gl::Buffer>() },
	m_tangentBuffer{ std::make_shared<gl::Buffer>() },
	m_uvBuffer{ std::make_shared<gl::Buffer>() }
{
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_POSITION, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_NORMAL, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_TANGENT, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_UV, 2, GL_FLOAT);

	Assimp::Importer importer;
	const aiScene* data{ importer.ReadFile(t_file, t_flags) };

	STHE_ASSERT(data != nullptr, "Failed to read file");
	
	unsigned int indexCount{ 0 };
	unsigned int vertexCount{ 0 };
	bool hasNormals{ false };
	bool hasTangents{ false };
	bool hasUVs{ false };

	for (unsigned int i{ 0 }; i < data->mNumMeshes; ++i)
	{
		const aiMesh& subMeshData{ *data->mMeshes[i] };

		STHE_ASSERT(subMeshData.HasFaces() && subMeshData.HasPositions(), "Mesh must have faces and positions");
		STHE_ASSERT(subMeshData.mPrimitiveTypes == static_cast<unsigned int>(aiPrimitiveType_TRIANGLE), "Primitive must be a triangle");

		indexCount += 3 * subMeshData.mNumFaces;
		vertexCount += subMeshData.mNumVertices;
		hasNormals |= subMeshData.HasNormals();
		hasTangents |= subMeshData.HasTangentsAndBitangents();
		hasUVs |= subMeshData.HasTextureCoords(0);
	}

	m_subMeshes.reserve(data->mNumMeshes);
	m_indices.reserve(indexCount);
	m_positions.reserve(vertexCount);
	m_hasChanged.set(STHE_MESH_INDEX_BIT);
	m_hasChanged.set(STHE_MESH_POSITION_BIT);

	if (hasNormals)
	{
		m_normals.reserve(vertexCount);
		m_hasChanged.set(STHE_MESH_NORMAL_BIT);
	}

	if (hasTangents)
	{
		m_tangents.reserve(vertexCount);
		m_hasChanged.set(STHE_MESH_TANGENT_BIT);
	}

	if (hasUVs)
	{
		m_uvs.reserve(vertexCount);
		m_hasChanged.set(STHE_MESH_UV_BIT);
	}

	vertexCount = 0;

	for (unsigned int i{ 0 }; i < data->mNumMeshes; ++i)
	{
		const aiMesh& subMeshData{ *data->mMeshes[i] };

		m_subMeshes.emplace_back(getIndexCount(), 3 * static_cast<int>(subMeshData.mNumFaces));

		for (unsigned int j{ 0 }; j < subMeshData.mNumFaces; ++j)
		{
			const aiFace& face{ subMeshData.mFaces[j] };
			m_indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[0]));
			m_indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[1]));
			m_indices.emplace_back(static_cast<int>(vertexCount + face.mIndices[2]));
		}

		for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
		{
			const aiVector3D& position{ subMeshData.mVertices[j] };
			m_positions.emplace_back(position.x, position.y, position.z, 1.0f);
		}

		vertexCount += subMeshData.mNumVertices;

		if (subMeshData.HasNormals())
		{
			for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
			{
				const aiVector3D& normal{ subMeshData.mNormals[j] };
				m_normals.emplace_back(normal.x, normal.y, normal.z, 0.0f);
			}
		}
		else
		{
			m_normals.resize(vertexCount, glm::vec4{ 0.0f });
		}

		if (subMeshData.HasTangentsAndBitangents())
		{
			for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
			{
				const aiVector3D& tangent{ subMeshData.mTangents[j] };
				m_tangents.emplace_back(tangent.x, tangent.y, tangent.z, 0.0f);
			}
		}
		else
		{
			m_tangents.resize(vertexCount, glm::vec4{ 0.0f });
		}

		if (subMeshData.HasTextureCoords(0))
		{
			for (unsigned int j{ 0 }; j < subMeshData.mNumVertices; ++j)
			{
				const aiVector3D& uv{ subMeshData.mTextureCoords[0][j] };
				m_uvs.emplace_back(uv.x, uv.y);
			}
		}
		else
		{
			m_uvs.resize(vertexCount, glm::vec2{ 0.0f });
		}
	}

	upload();
}

Mesh::Mesh(const Mesh& t_mesh) noexcept :
	m_indices{ t_mesh.m_indices },
	m_positions{ t_mesh.m_positions },
	m_normals{ t_mesh.m_normals },
	m_tangents{ t_mesh.m_tangents },
	m_uvs{ t_mesh.m_uvs },
	m_hasChanged{ t_mesh.m_hasChanged },
	m_vertexArray{ std::make_shared<gl::VertexArray>() },
	m_indexBuffer{ std::make_shared<gl::Buffer>(*t_mesh.m_indexBuffer) },
	m_positionBuffer{ std::make_shared<gl::Buffer>(*t_mesh.m_positionBuffer) },
	m_normalBuffer{ std::make_shared<gl::Buffer>(*t_mesh.m_normalBuffer) },
	m_tangentBuffer{ std::make_shared<gl::Buffer>(*t_mesh.m_tangentBuffer) },
	m_uvBuffer{ std::make_shared<gl::Buffer>(*t_mesh.m_uvBuffer) }
{
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_POSITION, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_NORMAL, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_TANGENT, 4, GL_FLOAT);
	m_vertexArray->setVertexAttributeFormat(STHE_VERTEX_ATTRIBUTE_UV, 2, GL_FLOAT);
	update();
}

// Operator
Mesh& Mesh::operator=(const Mesh& t_mesh) noexcept
{
	if (this != &t_mesh)
	{
		m_indices = t_mesh.m_indices;
		m_positions = t_mesh.m_positions;
		m_normals = t_mesh.m_normals;
		m_tangents = t_mesh.m_tangents;
		m_uvs = t_mesh.m_uvs;
		m_hasChanged = t_mesh.m_hasChanged; 
		*m_indexBuffer = *t_mesh.m_indexBuffer;
		*m_positionBuffer = *t_mesh.m_positionBuffer;
		*m_normalBuffer = *t_mesh.m_normalBuffer;
		*m_tangentBuffer = *t_mesh.m_tangentBuffer;
		*m_uvBuffer = *t_mesh.m_uvBuffer;
		update();
	}
	
	return *this;
}

// Functionality
void Mesh::bind() const
{
	m_vertexArray->bind();
}

void Mesh::addSubMesh(const SubMesh& t_subMesh)
{
	m_subMeshes.emplace_back(t_subMesh);
}

void Mesh::removeSubMesh(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < getSubMeshCount(), "Index must refer to an existing sub mesh");

	m_subMeshes.erase(m_subMeshes.begin() + t_index);
}

void Mesh::upload()
{
	if (m_hasChanged.test(STHE_MESH_INDEX_BIT))
	{
		m_indexBuffer->reinitialize(m_indices);
		m_hasChanged.reset(STHE_MESH_INDEX_BIT);

		if (m_indexBuffer->hasStorage())
		{
			m_vertexArray->attachIndexBuffer(*m_indexBuffer);
		}
	}

	if (m_hasChanged.test(STHE_MESH_POSITION_BIT))
	{
		m_positionBuffer->reinitialize(m_positions);
		m_hasChanged.reset(STHE_MESH_POSITION_BIT);

		if (m_positionBuffer->hasStorage())
		{
			m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_POSITION, *m_positionBuffer);
			m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_POSITION);
		}
		else
		{
			m_vertexArray->disableVertexAttribute(STHE_VERTEX_ATTRIBUTE_POSITION);
		}
	}

	if (m_hasChanged.test(STHE_MESH_NORMAL_BIT))
	{
		m_normalBuffer->reinitialize(m_normals);
		m_hasChanged.reset(STHE_MESH_NORMAL_BIT);

		if (m_normalBuffer->hasStorage())
		{
			m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_NORMAL, *m_normalBuffer);
			m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_NORMAL);
		}
		else
		{
			m_vertexArray->disableVertexAttribute(STHE_VERTEX_ATTRIBUTE_NORMAL);
		}
	}

	if (m_hasChanged.test(STHE_MESH_TANGENT_BIT))
	{
		m_tangentBuffer->reinitialize(m_tangents);
		m_hasChanged.reset(STHE_MESH_TANGENT_BIT);

		if (m_tangentBuffer->hasStorage())
		{
			m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_TANGENT, *m_tangentBuffer);
			m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_TANGENT);
		}
		else
		{
			m_vertexArray->disableVertexAttribute(STHE_VERTEX_ATTRIBUTE_TANGENT);
		}
	}

	if (m_hasChanged.test(STHE_MESH_UV_BIT))
	{
		m_uvBuffer->reinitialize(m_uvs);
		m_hasChanged.reset(STHE_MESH_UV_BIT);

		if (m_uvBuffer->hasStorage())
		{
			m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_UV, *m_uvBuffer);
			m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_UV);
		}
		else
		{
			m_vertexArray->disableVertexAttribute(STHE_VERTEX_ATTRIBUTE_UV);
		}
	}
}

void Mesh::clear()
{
	m_subMeshes.clear();
	m_indices.clear();
	m_positions.clear();
	m_normals.clear();
	m_tangents.clear();
	m_uvs.clear();
	m_hasChanged.set();
}

void Mesh::update() const
{
	if (m_indexBuffer->hasStorage())
	{
		m_vertexArray->attachIndexBuffer(*m_indexBuffer);
	}

	if (m_positionBuffer->hasStorage())
	{
		m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_POSITION, *m_positionBuffer);
		m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_POSITION);
	}

	if (m_normalBuffer->hasStorage())
	{
		m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_NORMAL, *m_normalBuffer);
		m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_NORMAL);
	}

	if (m_tangentBuffer->hasStorage())
	{
		m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_TANGENT, *m_tangentBuffer);
		m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_TANGENT);
	}

	if (m_uvBuffer->hasStorage())
	{
		m_vertexArray->attachVertexBuffer(STHE_VERTEX_ATTRIBUTE_UV, *m_uvBuffer);
		m_vertexArray->enableVertexAttribute(STHE_VERTEX_ATTRIBUTE_UV);
	}
}

// Setters
void Mesh::setSubMesh(const int t_index, const SubMesh& t_subMesh)
{
	STHE_ASSERT(t_index >= 0 && t_index < getSubMeshCount(), "Index must refer to an existing sub mesh");
	
	m_subMeshes[t_index] = t_subMesh;
}

void Mesh::setSubMeshes(const std::vector<SubMesh>& t_subMeshes)
{
	m_subMeshes = t_subMeshes;
}

void Mesh::setIndices(const std::vector<int>& t_indices)
{
	m_indices = t_indices;
	m_hasChanged.set(STHE_MESH_INDEX_BIT);
}

void Mesh::setPositions(const std::vector<glm::vec3>& t_positions)
{
	m_positions.resize(t_positions.size());

	for (size_t i{ 0 }; i < m_positions.size(); ++i)
	{
		m_positions[i] = glm::vec4{ t_positions[i], 1.0f };
	}

	m_hasChanged.set(STHE_MESH_POSITION_BIT);
}

void Mesh::setPositions(const std::vector<glm::vec4>& t_positions)
{
	m_positions = t_positions;
	m_hasChanged.set(STHE_MESH_POSITION_BIT);
}

void Mesh::setNormals(const std::vector<glm::vec3>& t_normals)
{
	m_normals.resize(t_normals.size());

	for (size_t i{ 0 }; i < t_normals.size(); ++i)
	{
		m_normals[i] = glm::vec4{ t_normals[i], 0.0f };
	}

	m_hasChanged.set(STHE_MESH_NORMAL_BIT);
}

void Mesh::setNormals(const std::vector<glm::vec4>& t_normals)
{
	m_normals = t_normals;
	m_hasChanged.set(STHE_MESH_NORMAL_BIT);
}

void Mesh::setTangents(const std::vector<glm::vec3>& t_tangents)
{
	m_tangents.resize(t_tangents.size());

	for (size_t i{ 0 }; i < t_tangents.size(); ++i)
	{
		m_tangents[i] = glm::vec4{ t_tangents[i], 0.0f };
	}

	m_hasChanged.set(STHE_MESH_TANGENT_BIT);
}

void Mesh::setTangents(const std::vector<glm::vec4>& t_tangents)
{
	m_tangents = t_tangents;
	m_hasChanged.set(STHE_MESH_TANGENT_BIT);
}

void Mesh::setUVs(const std::vector<glm::vec2>& t_uvs)
{
	m_uvs = t_uvs;
	m_hasChanged.set(STHE_MESH_UV_BIT);
}

// Getters
const SubMesh& Mesh::getSubMesh(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < getSubMeshCount(), "Index must refer to an existing sub mesh");

	return m_subMeshes[t_index];
}

const std::vector<SubMesh>& Mesh::getSubMeshes() const
{
	return m_subMeshes;
}

const std::vector<int>& Mesh::getIndices() const
{
	return m_indices;
}

const std::vector<glm::vec4>& Mesh::getPositions() const
{
	return m_positions;
}

const std::vector<glm::vec4>& Mesh::getNormals() const
{
	return m_normals;
}

const std::vector<glm::vec4>& Mesh::getTangents() const
{
	return m_tangents;
}

const std::vector<glm::vec2>& Mesh::getUVs() const
{
	return m_uvs;
}

int Mesh::getSubMeshCount() const
{
	return static_cast<int>(m_subMeshes.size());
}

int Mesh::getIndexCount() const
{
	return static_cast<int>(m_indices.size());
}

int Mesh::getVertexCount() const
{
	return static_cast<int>(m_positions.size());
}

const std::shared_ptr<gl::VertexArray>& Mesh::getVertexArray() const
{
	return m_vertexArray;
}

const std::shared_ptr<gl::Buffer>& Mesh::getIndexBuffer() const
{
	return m_indexBuffer;
}

const std::shared_ptr<gl::Buffer>& Mesh::getPositionBuffer() const
{
	return m_positionBuffer;
}

const std::shared_ptr<gl::Buffer>& Mesh::getNormalBuffer() const
{
	return m_normalBuffer;
}

const std::shared_ptr<gl::Buffer>& Mesh::getTangentBuffer() const
{
	return m_tangentBuffer;
}

const std::shared_ptr<gl::Buffer>& Mesh::getUVBuffer() const
{
	return m_uvBuffer;
}

bool Mesh::hasSubMeshes() const
{
	return !m_subMeshes.empty();
}

bool Mesh::hasIndices() const
{
	return !m_indices.empty();
}

bool Mesh::hasPositions() const
{
	return !m_positions.empty();
}

bool Mesh::hasNormals() const
{
	return !m_normals.empty();
}

bool Mesh::hasTangents() const
{
	return !m_tangents.empty();
}

bool Mesh::hasUVs() const
{
	return !m_uvs.empty();
}

}
