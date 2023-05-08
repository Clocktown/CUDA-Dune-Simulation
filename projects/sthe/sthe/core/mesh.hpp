#pragma once

#include "sub_mesh.hpp"
#include <sthe/gl/vertex_array.hpp>
#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <assimp/postprocess.h>
#include <memory>
#include <vector>
#include <bitset>

namespace sthe
{

namespace ScreenFillingQuad
{
    void draw();
}

enum class Geometry : unsigned char
{
	Quad,
	Plane,
	Cube,
	Cylinder,
	Cone,
	Sphere
};

class Mesh
{
public:
	// Constructors
	Mesh();
	explicit Mesh(const Geometry t_geometry);
	explicit Mesh(const std::string& t_file, const unsigned int t_flags = 0);
	Mesh(const Mesh& t_mesh) noexcept;
	Mesh(Mesh&& t_mesh) = default;

	// Destructor
	~Mesh() = default;

	// Operators
	Mesh& operator=(const Mesh& t_mesh) noexcept;
	Mesh& operator=(Mesh&& t_mesh) = default;

	// Functionality
	void bind() const;
	void addSubMesh(const SubMesh& t_subMesh);
	void removeSubMesh(const int t_index);
	void upload();
	void clear();

	// Setters
	void setSubMesh(const int t_index, const SubMesh& t_subMesh);
	void setSubMeshes(const std::vector<SubMesh>& t_subMeshes);
	void setIndices(const std::vector<int>& t_indices);
	void setPositions(const std::vector<glm::vec3>& t_positions);
	void setPositions(const std::vector<glm::vec4>& t_positions);
	void setNormals(const std::vector<glm::vec3>& t_normals);
	void setNormals(const std::vector<glm::vec4>& t_normals);
	void setTangents(const std::vector<glm::vec3>& t_tangents);
	void setTangents(const std::vector<glm::vec4>& t_tangents);
	void setUVs(const std::vector<glm::vec2>& t_uvs);
	 
	// Getters
	const SubMesh& getSubMesh(const int t_index) const;
	const std::vector<SubMesh>& getSubMeshes() const;
	const std::vector<int>& getIndices() const;
	const std::vector<glm::vec4>& getPositions() const;
	const std::vector<glm::vec4>& getNormals() const;
	const std::vector<glm::vec4>& getTangents() const;
	const std::vector<glm::vec2>& getUVs() const;
	int getSubMeshCount() const;
	int getIndexCount() const;
	int getVertexCount() const;
	const std::shared_ptr<gl::VertexArray>& getVertexArray() const;
	const std::shared_ptr<gl::Buffer>& getIndexBuffer() const;
	const std::shared_ptr<gl::Buffer>& getPositionBuffer() const;
	const std::shared_ptr<gl::Buffer>& getNormalBuffer() const;
	const std::shared_ptr<gl::Buffer>& getTangentBuffer() const;
	const std::shared_ptr<gl::Buffer>& getUVBuffer() const;
	bool hasSubMeshes() const;
	bool hasIndices() const;
	bool hasPositions() const;
	bool hasNormals() const;
	bool hasTangents() const;
	bool hasUVs() const;
private:
	// Functionality 
	void update() const;

	// Attributes
	std::vector<SubMesh> m_subMeshes;
	std::vector<int> m_indices;
	std::vector<glm::vec4> m_positions;
	std::vector<glm::vec4> m_normals;
	std::vector<glm::vec4> m_tangents;
	std::vector<glm::vec2> m_uvs;
	std::bitset<5> m_hasChanged;

	std::shared_ptr<gl::VertexArray> m_vertexArray;
	std::shared_ptr<gl::Buffer> m_indexBuffer;
	std::shared_ptr<gl::Buffer> m_positionBuffer;
	std::shared_ptr<gl::Buffer> m_normalBuffer;
	std::shared_ptr<gl::Buffer> m_tangentBuffer;
	std::shared_ptr<gl::Buffer> m_uvBuffer;
};

}
