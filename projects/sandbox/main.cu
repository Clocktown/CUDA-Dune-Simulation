#include <sthe/cu/buffer.hpp>
#include <sthe/cu/array2d.hpp>
#include <sthe/cu/stopwatch.hpp>
#include <sthe/device/buffer.cuh>
#include <sthe/device/array2d.cuh>
#include <sthe/device/vector_extension.cuh>
#include <device_launch_parameters.h>

#define THREADS_1D 512
#define THREADS_2D 8
#define WIDTH 4096
#define HEIGHT 4096
#define COUNT (WIDTH * HEIGHT)
#define ITERS 1000

using namespace sthe;

__global__ void gaussBuffer1D(const device::Buffer<float4> t_in, device::Buffer<float4> t_out)
{
	const int idx{ static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) };
	const int x{ idx % WIDTH };
	const int y{ idx / WIDTH };

	float4 sum{ make_float4(0.0f) };

	for (int i{ -1 }; i <= 1; ++i)
	{
		for (int j{ -1 }; j <= 1; ++j)
		{
			const int2 cell{ (x + j), (y + i) };

			if (cell.x < 0 || cell.y < 0 || cell.x >= WIDTH || cell.y >= HEIGHT)
			{
				continue;
			}

			sum += t_in[cell.x + WIDTH * cell.y];
		}
	}

	sum /= 9.0f;

	t_out[x + WIDTH * y] = sum;
}

__global__ void gaussBuffer2D(const device::Buffer<float4> t_in, device::Buffer<float4> t_out)
{
	const int x{ static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) };
	const int y{ static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y) };

	float4 sum{ make_float4(0.0f) };

	for (int i{ -1 }; i <= 1; ++i)
	{
		for (int j{ -1 }; j <= 1; ++j)
		{
			const int2 cell{ (x + j), (y + i) };

			if (cell.x < 0 || cell.y < 0 || cell.x >= WIDTH || cell.y >= HEIGHT)
			{
				continue;
			}

			sum += t_in[cell.x + WIDTH * cell.y];
		}
	}

	sum /= 9.0f;

	t_out[x + WIDTH * y] = sum;
}

__global__ void gaussArray2D(const device::Array2D<float4> t_in, device::Array2D<float4> t_out)
{
	const int2 idx{ static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x),
	                static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y) };

	float4 sum{ make_float4(0.0f) };

	for (int i{ -1 }; i <= 1; ++i)
	{
		for (int j{ -1 }; j <= 1; ++j)
		{
			const int2 cell{ idx + make_int2(j, i) };

			sum += t_in.read(cell, cudaBoundaryModeZero);
			//sum += t_in.sample(make_float2(cell) + 0.5f);
		}
	}

	sum /= 9.0f;

	t_out.write(idx, sum);
}

void main()
{
	cudaTextureDesc tex{};
	tex.filterMode = cudaFilterModePoint;
	tex.addressMode[0] = cudaAddressModeBorder;
	tex.addressMode[1] = cudaAddressModeBorder;
	tex.normalizedCoords = 0;

	cu::Buffer buffer1(COUNT, sizeof(float4));
	cu::Buffer buffer2(COUNT, sizeof(float4));
	cu::Array2D array1(WIDTH, HEIGHT, cudaCreateChannelDesc<float4>());
	cu::Array2D array2(WIDTH, HEIGHT, cudaCreateChannelDesc<float4>());
	device::Array2D<float4> deviceArray1;
	device::Array2D<float4> deviceArray2;
	deviceArray1.texture = array1.recreateTexture(tex);
	deviceArray1.surface = array1.recreateSurface();
	deviceArray2.texture = array2.recreateTexture(tex);
	deviceArray2.surface = array2.recreateSurface();

	const unsigned int blocks1D{ COUNT / THREADS_1D };
	const dim3 blocks2D{ WIDTH / THREADS_2D, HEIGHT / THREADS_2D };

	cu::Stopwatch sw;
	float timeBuffer1D{ 0.0f };
	float timeBuffer2D{ 0.0f };
	float timeArray2D{ 0.0f };

	for (int i{ 0 }; i < 100; ++i)
	{
		gaussBuffer1D<<<blocks1D, THREADS_1D>>> (buffer1.getData<float4>(), buffer2.getData<float4>());
		gaussBuffer2D<<<blocks2D, dim3{ THREADS_2D, THREADS_2D }>>>(buffer1.getData<float4>(), buffer2.getData<float4>());
		gaussArray2D<<<blocks2D, dim3{ THREADS_2D, THREADS_2D }>>>(deviceArray1, deviceArray2);
	}

	for (int i{ 0 }; i < ITERS; ++i)
	{
		sw.start();
		gaussBuffer1D<<<blocks1D, THREADS_1D>>>(buffer1.getData<float4>(), buffer2.getData<float4>());
		sw.stop();
		timeBuffer1D += sw.getTime();

		sw.start();
		gaussBuffer2D<<<blocks2D, dim3{ THREADS_2D, THREADS_2D }>>>(buffer1.getData<float4>(), buffer2.getData<float4>());
		sw.stop();
		timeBuffer2D += sw.getTime();

		sw.start();
		gaussArray2D<<<blocks2D, dim3{ THREADS_2D, THREADS_2D }>>>(deviceArray1, deviceArray2);
		sw.stop();
		timeArray2D += sw.getTime();
	}

	printf("Time Buffer 1D: %f\n", timeBuffer1D / ITERS);
	printf("Time Buffer 2D: %f\n", timeBuffer2D / ITERS);
	printf("Time Array 2D:  %f\n", timeArray2D / ITERS);
}
