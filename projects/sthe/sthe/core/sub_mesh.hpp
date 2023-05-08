#pragma once

#include <glad/glad.h>

namespace sthe
{

class SubMesh
{
public:
	// Constructors
	SubMesh();
	SubMesh(const int t_firstIndex, const int t_indexCount, const GLenum t_drawMode = GL_TRIANGLES);
	SubMesh(const SubMesh& t_subMesh) = default;
	SubMesh(SubMesh&& t_subMesh) = default;

	// Destructor
	~SubMesh() = default;

	// Operators
	SubMesh& operator=(const SubMesh& t_subMesh) = default;
	SubMesh& operator=(SubMesh&& t_subMesh) = default;

	// Setters
	void setFirstIndex(const int t_firstIndex);
	void setIndexCount(const int t_indexCount);
	void setDrawMode(const GLenum t_drawMode);

	// Getters
	int getFirstIndex() const;
	int getIndexCount() const;
	GLenum getDrawMode() const;
private:
	// Attributes
	int m_firstIndex;
	int m_indexCount;
	GLenum m_drawMode;
};

}
