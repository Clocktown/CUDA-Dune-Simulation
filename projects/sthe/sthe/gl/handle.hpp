#pragma once

#include <glad/glad.h>

namespace sthe
{
namespace gl
{

class Handle
{
public:
	// Constructors
	Handle() = default;
	Handle(const Handle& t_handle) = delete;
	Handle(Handle&& t_handle) = delete;

	// Destructor
	virtual ~Handle() = default;

	// Operators
	Handle& operator=(const Handle& t_handle) = delete;
	Handle& operator=(Handle&& t_handle) = delete;

	// Getter
	virtual GLuint getHandle() const = 0;
};

}
}
