#include "sub_mesh.hpp"
#include <sthe/config/debug.hpp>

namespace sthe
{

// Constructors
SubMesh::SubMesh() :
	m_firstIndex{ 0 },
	m_indexCount{ 0 },
	m_drawMode{ GL_NONE }
{

}

SubMesh::SubMesh(const int t_firstIndex, const int t_indexCount, const GLenum t_drawMode) :
	m_firstIndex { t_firstIndex },
    m_indexCount{ t_indexCount },
	m_drawMode{ t_drawMode }
{
	STHE_ASSERT(t_firstIndex >= 0, "First index must be greater than or equal to 0");
	STHE_ASSERT(t_indexCount >= 0, "Index count must be greater than or equal to 0");
}

// Setters
void SubMesh::setFirstIndex(const int t_firstIndex)
{
	STHE_ASSERT(t_firstIndex >= 0, "First index must be greater than or equal to 0");

	m_firstIndex = t_firstIndex;
}

void SubMesh::setIndexCount(const int t_indexCount)
{
	STHE_ASSERT(t_indexCount >= 0, "Index count must be greater than or equal to 0");

	m_indexCount = t_indexCount;
}

void SubMesh::setDrawMode(const GLenum t_drawMode)
{
	m_drawMode = t_drawMode;
}

// Getters
int SubMesh::getFirstIndex() const
{
	return m_firstIndex;
}

int SubMesh::getIndexCount() const
{
	return m_indexCount;
}

GLenum SubMesh::getDrawMode() const
{
	return m_drawMode;
}

}
