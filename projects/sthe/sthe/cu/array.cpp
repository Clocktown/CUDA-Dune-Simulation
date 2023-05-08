#include "array.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace sthe
{
namespace cu
{

std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, cudaChannelFormatDesc& t_format)
{
	int channels;
	std::shared_ptr<unsigned char> source{ sthe::readImage2D(t_file, t_width, t_height, channels) };

	switch (channels)
	{
	case 1:
		t_format = cudaCreateChannelDesc<unsigned char>();
		break;
	case 2:
		t_format = cudaCreateChannelDesc<uchar2>();
		break;
	case 3:
	{
		const int pixelCount{ t_width * t_height };
		unsigned char* const modifiedSource{ new unsigned char[4 * pixelCount] };

		for (int i{ 0 }, j{ 0 }; i < 4 * pixelCount; i += 4)
		{
			modifiedSource[i] = source.get()[j++];
			modifiedSource[i + 1] = source.get()[j++];
			modifiedSource[i + 2] = source.get()[j++];
			modifiedSource[i + 3] = 255;
		}

		source.reset(modifiedSource, std::default_delete<unsigned char[]>());
	}
	case 4:
		t_format = cudaCreateChannelDesc<uchar4>();
		break;
	default:
		STHE_ERROR("Channels must be between 1 and 4");

		break;
	}

	return source;
}

// Getters
int Array::getWidth() const
{
	return 0;
}

int Array::getHeight() const
{
	return 0;
}

int Array::getDepth() const
{
	return 0;
}

int Array::getStride() const
{
	const cudaChannelFormatDesc& format{ getFormat() };
	const int bits{ format.x + format.y + format.z + format.w };

	STHE_ASSERT(bits % 8 == 0, "Format is not valid");

	return bits / 8;
}

bool Array::hasStorage() const
{
	return getWidth() > 0;
}

}
}
