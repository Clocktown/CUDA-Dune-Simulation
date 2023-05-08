#pragma once

namespace sthe
{
namespace gl
{

struct IndirectCommand
{
    int indexCount;
    int instanceCount;
    int firstIndex;
    int baseVertex;
    int baseInstance;
};

}
}
