#pragma once

#include "function_qualifier.cuh"
#include <vector_types.h>
#include <vector_functions.h>

#ifndef __CUDACC__
#   include <cmath>

	CU_INLINE CU_HOST_DEVICE int min(const int a, const int b)
	{
		return a < b ? a : b;
	}

	CU_INLINE CU_HOST_DEVICE unsigned int min(const unsigned int a, const unsigned int b)
	{
		return a < b ? a : b;
	}

	CU_INLINE CU_HOST_DEVICE int max(const int a, const int b)
	{
		return a > b ? a : b;
	}

	CU_INLINE CU_HOST_DEVICE unsigned int max(const unsigned int a, const unsigned int b)
	{
		return a > b ? a : b;
	}

	CU_INLINE CU_HOST_DEVICE float rsqrtf(const float a)
	{
		return 1.0f / sqrtf(a);
	}
#endif

CU_INLINE CU_HOST_DEVICE int sign(const int a)
{
	return a >= 0 ? 1 : -1;
}

CU_INLINE CU_HOST_DEVICE float sign(const float a)
{
	return a >= 0.0f ? 1.0f : -1.0f;
}

CU_INLINE CU_HOST_DEVICE int clamp(const int a, const int x, const int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE unsigned int clamp(const unsigned int a, const unsigned int x, const unsigned int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE float clamp(const float a, const float x, const float y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float lerp(const float a, const float b, const float t)
{
	return a + t * (b - a);
}

CU_INLINE CU_HOST_DEVICE float bilerp(const float a00, const float a10, const float a01, const float a11, const float u, const float v)
{
	return lerp(lerp(a00, a10, u), lerp(a01, a11, u), v);
}

CU_INLINE CU_HOST_DEVICE int2 make_int2(const uint2& a)
{
	return int2{ static_cast<int>(a.x), static_cast<int>(a.y) };
}

CU_INLINE CU_HOST_DEVICE int2 make_int2(const float2& a)
{
	return int2{ static_cast<int>(a.x), static_cast<int>(a.y) };
}

CU_INLINE CU_HOST_DEVICE int2 make_int2(const int a)
{
	return int2{ a, a };
}

CU_INLINE CU_HOST_DEVICE int2 make_int2(const int3& a)
{
	return int2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE int2 make_int2(const int4& a)
{
	return int2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const int2& a, const int2& b)
{
	return a.x == b.x && a.y == b.y;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const int2& a, const int2& b)
{
	return a.x != b.x || a.y != b.y;
}

CU_INLINE CU_HOST_DEVICE void operator++(int2& a)
{
	a.x++;
	a.y++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int2& a, const int b)
{
	a.x += b;
	a.y += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int2& a, const int2& b)
{
	a.x += b.x;
	a.y += b.y;
}

CU_INLINE CU_HOST_DEVICE int2 operator+(const int2& a, const int b)
{
	return int2{ a.x + b, a.y + b };
}

CU_INLINE CU_HOST_DEVICE int2 operator+(const int a, const int2& b)
{
	return int2{ a + b.x, a + b.y };
}

CU_INLINE CU_HOST_DEVICE int2 operator+(const int2& a, const int2& b)
{
	return int2{ a.x + b.x, a.y + b.y };
}

CU_INLINE CU_HOST_DEVICE void operator--(int2& a)
{
	a.x--;
	a.y--;
}

CU_INLINE CU_HOST_DEVICE int2 operator-(const int2& a)
{
	return int2{ -a.x, -a.y };
}

CU_INLINE CU_HOST_DEVICE void operator-=(int2& a, const int b)
{
	a.x -= b;
	a.y -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(int2& a, const int2& b)
{
	a.x -= b.x;
	a.y -= b.y;
}

CU_INLINE CU_HOST_DEVICE int2 operator-(const int2& a, const int b)
{
	return int2{ a.x - b, a.y - b };
}

CU_INLINE CU_HOST_DEVICE int2 operator-(const int a, const int2& b)
{
	return int2{ a - b.x, a - b.y };
}

CU_INLINE CU_HOST_DEVICE int2 operator-(const int2& a, const int2& b)
{
	return int2{ a.x - b.x, a.y - b.y };
}

CU_INLINE CU_HOST_DEVICE void operator*=(int2& a, const int b)
{
	a.x *= b;
	a.y *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(int2& a, const int2& b)
{
	a.x *= b.x;
	a.y *= b.y;
}

CU_INLINE CU_HOST_DEVICE int2 operator*(const int2& a, const int b)
{
	return int2{ a.x * b, a.y * b };
}

CU_INLINE CU_HOST_DEVICE int2 operator*(const int a, const int2& b)
{
	return int2{ a * b.x, a * b.y };
}

CU_INLINE CU_HOST_DEVICE int2 operator*(const int2& a, const int2& b)
{
	return int2{ a.x * b.x, a.y * b.y };
}

CU_INLINE CU_HOST_DEVICE void operator/=(int2& a, const int b)
{
	a.x /= b;
	a.y /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(int2& a, const int2& b)
{
	a.x /= b.x;
	a.y /= b.y;
}

CU_INLINE CU_HOST_DEVICE int2 operator/(const int2& a, const int b)
{
	return int2{ a.x / b, a.y / b };
}

CU_INLINE CU_HOST_DEVICE int2 operator/(const int a, const int2& b)
{
	return int2{ a / b.x, a / b.y };
}

CU_INLINE CU_HOST_DEVICE int2 operator/(const int2& a, const int2& b)
{
	return int2{ a.x / b.x, a.y / b.y };
}

CU_INLINE CU_HOST_DEVICE int2 abs(const int2& a)
{
	return int2{ abs(a.x), abs(a.y) };
}

CU_INLINE CU_HOST_DEVICE int2 min(const int2& a, const int b)
{
	return int2{ min(a.x, b), min(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE int2 min(const int a, const int2& b)
{
	return int2{ min(a, b.x), min(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE int2 min(const int2& a, const int2& b)
{
	return int2{ min(a.x, b.x), min(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE int2 max(const int2& a, const int b)
{
	return int2{ max(a.x, b), max(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE int2 max(const int a, const int2& b)
{
	return int2{ max(a, b.x), max(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE int2 max(const int2& a, const int2& b)
{
	return int2{ max(a.x, b.x), max(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE int2 sign(const int2& a)
{
	return int2{ sign(a.x), sign(a.y) };
}

CU_INLINE CU_HOST_DEVICE int2 clamp(const int2& a, const int x, const int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int2 clamp(const int2& a, const int2& x, const int2& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int dot(const int2& a, const int2& b)
{
	return a.x * b.x + a.y * b.y;
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const uint3& a)
{
	return int3{ static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z) };
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const float3& a)
{
	return int3{ static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z) };
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const int a)
{
	return int3{ a, a, a };
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const int2& a, const int b)
{
	return int3{ a.x, a.y, b };
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const int a, const int2& b)
{
	return int3{ a, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE int3 make_int3(const int4& a)
{
	return int3{ a.x, a.y, a.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const int3& a, const int3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const int3& a, const int3& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

CU_INLINE CU_HOST_DEVICE void operator++(int3& a)
{
	a.x++;
	a.y++;
	a.z++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int3& a, const int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int3& a, const int3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

CU_INLINE CU_HOST_DEVICE int3 operator+(const int3& a, const int b)
{
	return int3{ a.x + b, a.y + b, a.z + b };
}

CU_INLINE CU_HOST_DEVICE int3 operator+(const int a, const int3& b)
{
	return int3{ a + b.x, a + b.y, a + b.z };
}

CU_INLINE CU_HOST_DEVICE int3 operator+(const int3& a, const int3& b)
{
	return int3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

CU_INLINE CU_HOST_DEVICE void operator--(int3& a)
{
	a.x--;
	a.y--;
	a.z--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(int3& a, const int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(int3& a, const int3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

CU_INLINE CU_HOST_DEVICE int3 operator-(const int3& a)
{
	return int3{ -a.x, -a.y, -a.z };
}

CU_INLINE CU_HOST_DEVICE int3 operator-(const int3& a, const int b)
{
	return int3{ a.x - b, a.y - b, a.z - b };
}

CU_INLINE CU_HOST_DEVICE int3 operator-(const int a, const int3& b)
{
	return int3{ a - b.x, a - b.y, a - b.z };
}

CU_INLINE CU_HOST_DEVICE int3 operator-(const int3& a, const int3& b)
{
	return int3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

CU_INLINE CU_HOST_DEVICE void operator*=(int3& a, const int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(int3& a, const int3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

CU_INLINE CU_HOST_DEVICE int3 operator*(const int3& a, const int b)
{
	return int3{ a.x * b, a.y * b, a.z * b };
}

CU_INLINE CU_HOST_DEVICE int3 operator*(const int a, const int3& b)
{
	return int3{ a * b.x, a * b.y, a * b.z };
}

CU_INLINE CU_HOST_DEVICE int3 operator*(const int3& a, const int3& b)
{
	return int3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

CU_INLINE CU_HOST_DEVICE void operator/=(int3& a, const int b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(int3& a, const int3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}

CU_INLINE CU_HOST_DEVICE int3 operator/(const int3& a, const int b)
{
	return int3{ a.x / b, a.y / b, a.z / b };
}

CU_INLINE CU_HOST_DEVICE int3 operator/(const int a, const int3& b)
{
	return int3{ a / b.x, a / b.y, a / b.z };
}

CU_INLINE CU_HOST_DEVICE int3 operator/(const int3& a, const int3& b)
{
	return int3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

CU_INLINE CU_HOST_DEVICE int3 abs(const int3& a)
{
	return int3{ abs(a.x), abs(a.y), abs(a.z) };
}

CU_INLINE CU_HOST_DEVICE int3 min(const int3& a, const int b)
{
	return int3{ min(a.x, b), min(a.y, b), min(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE int3 min(const int a, const int3& b)
{
	return int3{ min(a, b.x), min(a, b.y), min(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE int3 min(const int3& a, const int3& b)
{
	return int3{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE int3 max(const int3& a, const int b)
{
	return int3{ max(a.x, b), max(a.y, b), max(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE int3 max(const int a, const int3& b)
{
	return int3{ max(a, b.x), max(a, b.y), max(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE int3 max(const int3& a, const int3& b)
{
	return int3{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE int3 sign(const int3& a)
{
	return int3{ sign(a.x), sign(a.y), sign(a.z) };
}

CU_INLINE CU_HOST_DEVICE int3 clamp(const int3& a, const int x, const int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int3 clamp(const int3& a, const int3& x, const int3& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int dot(const int3& a, const int3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const uint4& a)
{
	return int4{ static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z), static_cast<int>(a.w) };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const float4& a)
{
	return int4{ static_cast<int>(a.x), static_cast<int>(a.y), static_cast<int>(a.z), static_cast<int>(a.w) };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int a)
{
	return int4{ a, a, a, a };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int2& a, const int b, const int c)
{
	return int4{ a.x, a.y, b, c };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int a, const int2& b, const int c)
{
	return int4{ a, b.x, b.y, c };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int a, const int b, const int2& c)
{
	return int4{ a, b, c.x, c.y };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int2& a, const int2& b)
{
	return int4{ a.x, a.y, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int3& a, const int b)
{
	return int4{ a.x, a.y, a.z, b };
}

CU_INLINE CU_HOST_DEVICE int4 make_int4(const int a, const int3& b)
{
	return int4{ a, b.x, b.y, b.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const int4& a, const int4& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const int4& a, const int4& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

CU_INLINE CU_HOST_DEVICE void operator++(int4& a)
{
	a.x++;
	a.y++;
	a.z++;
	a.w++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int4& a, const int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(int4& a, const int4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

CU_INLINE CU_HOST_DEVICE int4 operator+(const int4& a, const int b)
{
	return int4{ a.x + b, a.y + b, a.z + b, a.w + b };
}

CU_INLINE CU_HOST_DEVICE int4 operator+(const int a, const int4& b)
{
	return int4{ a + b.x, a + b.y, a + b.z, a + b.w };
}

CU_INLINE CU_HOST_DEVICE int4 operator+(const int4& a, const int4& b)
{
	return int4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

CU_INLINE CU_HOST_DEVICE void operator--(int4& a)
{
	a.x--;
	a.y--;
	a.z--;
	a.w--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(int4& a, const int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(int4& a, const int4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

CU_INLINE CU_HOST_DEVICE int4 operator-(const int4& a)
{
	return int4{ -a.x, -a.y, -a.z, -a.w };
}

CU_INLINE CU_HOST_DEVICE int4 operator-(const int4& a, const int b)
{
	return int4{ a.x - b, a.y - b, a.z - b, a.w - b };
}

CU_INLINE CU_HOST_DEVICE int4 operator-(const int a, const int4& b)
{
	return int4{ a - b.x, a - b.y, a - b.z, a - b.w };
}

CU_INLINE CU_HOST_DEVICE int4 operator-(const int4& a, const int4& b)
{
	return int4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

CU_INLINE CU_HOST_DEVICE void operator*=(int4& a, const int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(int4& a, const int4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

CU_INLINE CU_HOST_DEVICE int4 operator*(const int4& a, const int b)
{
	return int4{ a.x * b, a.y * b, a.z * b, a.w * b };
}

CU_INLINE CU_HOST_DEVICE int4 operator*(const int a, const int4& b)
{
	return int4{ a * b.x, a * b.y, a * b.z, a * b.w };
}

CU_INLINE CU_HOST_DEVICE int4 operator*(const int4& a, const int4& b)
{
	return int4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

CU_INLINE CU_HOST_DEVICE void operator/=(int4& a, const int b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(int4& a, const int4& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

CU_INLINE CU_HOST_DEVICE int4 operator/(const int4& a, const int b)
{
	return int4{ a.x / b, a.y / b, a.z / b, a.w / b };
}

CU_INLINE CU_HOST_DEVICE int4 operator/(const int a, const int4& b)
{
	return int4{ a / b.x, a / b.y, a / b.z, a / b.w };
}

CU_INLINE CU_HOST_DEVICE int4 operator/(const int4& a, const int4& b)
{
	return int4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

CU_INLINE CU_HOST_DEVICE int4 abs(const int4& a)
{
	return int4{ abs(a.x), abs(a.y), abs(a.z), abs(a.w) };
}

CU_INLINE CU_HOST_DEVICE int4 min(const int4& a, const int b)
{
	return int4{ min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE int4 min(const int a, const int4& b)
{
	return int4{ min(a, b.x), min(a, b.y), min(a, b.z), min(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE int4 min(const int4& a, const int4& b)
{
	return int4{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE int4 max(const int4& a, const int b)
{
	return int4{ max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE int4 max(const int a, const int4& b)
{
	return int4{ max(a, b.x), max(a, b.y), max(a, b.z), max(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE int4 max(const int4& a, const int4& b)
{
	return int4{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE int4 sign(const int4& a)
{
	return int4{ sign(a.x), sign(a.y), sign(a.z), sign(a.w) };
}

CU_INLINE CU_HOST_DEVICE int4 clamp(const int4& a, const int x, const int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int4 clamp(const int4& a, const int4& x, const int4& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE int dot(const int4& a, const int4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

CU_INLINE CU_HOST_DEVICE uint2 make_uint2(const int2& a)
{
	return uint2{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 make_uint2(const float2& a)
{
	return uint2{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 make_uint2(const unsigned int a)
{
	return uint2{ a, a };
}

CU_INLINE CU_HOST_DEVICE uint2 make_uint2(const uint3& a)
{
	return uint2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE uint2 make_uint2(const uint4& a)
{
	return uint2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const uint2& a, const uint2& b)
{
	return a.x == b.x && a.y == b.y;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const uint2& a, const uint2& b)
{
	return a.x != b.x || a.y != b.y;
}

CU_INLINE CU_HOST_DEVICE void operator++(uint2& a)
{
	a.x++;
	a.y++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint2& a, const unsigned int b)
{
	a.x += b;
	a.y += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint2& a, const uint2& b)
{
	a.x += b.x;
	a.y += b.y;
}

CU_INLINE CU_HOST_DEVICE uint2 operator+(const uint2& a, const unsigned int b)
{
	return uint2{ a.x + b, a.y + b };
}

CU_INLINE CU_HOST_DEVICE uint2 operator+(const unsigned int a, const uint2& b)
{
	return uint2{ a + b.x, a + b.y };
}

CU_INLINE CU_HOST_DEVICE uint2 operator+(const uint2& a, const uint2& b)
{
	return uint2{ a.x + b.x, a.y + b.y };
}

CU_INLINE CU_HOST_DEVICE void operator--(uint2& a)
{
	a.x--;
	a.y--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint2& a, const unsigned int b)
{
	a.x -= b;
	a.y -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint2& a, const uint2& b)
{
	a.x -= b.x;
	a.y -= b.y;
}

CU_INLINE CU_HOST_DEVICE uint2 operator-(const uint2& a, const unsigned int b)
{
	return uint2{ a.x - b, a.y - b };
}

CU_INLINE CU_HOST_DEVICE uint2 operator-(const unsigned int a, const uint2& b)
{
	return uint2{ a - b.x, a - b.y };
}

CU_INLINE CU_HOST_DEVICE uint2 operator-(const uint2& a, const uint2& b)
{
	return uint2{ a.x - b.x, a.y - b.y };
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint2& a, const unsigned int b)
{
	a.x *= b;
	a.y *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint2& a, const uint2& b)
{
	a.x *= b.x;
	a.y *= b.y;
}

CU_INLINE CU_HOST_DEVICE uint2 operator*(const uint2& a, const unsigned int b)
{
	return uint2{ a.x * b, a.y * b };
}

CU_INLINE CU_HOST_DEVICE uint2 operator*(const unsigned int a, const uint2& b)
{
	return uint2{ a * b.x, a * b.y };
}

CU_INLINE CU_HOST_DEVICE uint2 operator*(const uint2& a, const uint2& b)
{
	return uint2{ a.x * b.x, a.y * b.y };
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint2& a, const unsigned int b)
{
	a.x /= b;
	a.y /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint2& a, const uint2& b)
{
	a.x /= b.x;
	a.y /= b.y;
}

CU_INLINE CU_HOST_DEVICE uint2 operator/(const uint2& a, const unsigned int b)
{
	return uint2{ a.x / b, a.y / b };
}

CU_INLINE CU_HOST_DEVICE uint2 operator/(const unsigned int a, const uint2& b)
{
	return uint2{ a / b.x, a / b.y };
}

CU_INLINE CU_HOST_DEVICE uint2 operator/(const uint2& a, const uint2& b)
{
	return uint2{ a.x / b.x, a.y / b.y };
}

CU_INLINE CU_HOST_DEVICE uint2 min(const uint2& a, const unsigned int b)
{
	return uint2{ min(a.x, b), min(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE uint2 min(const unsigned int a, const uint2& b)
{
	return uint2{ min(a, b.x), min(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 min(const uint2& a, const uint2& b)
{
	return uint2{ min(a.x, b.x), min(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 max(const uint2& a, const unsigned int b)
{
	return uint2{ max(a.x, b), max(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE uint2 max(const unsigned int a, const uint2& b)
{
	return uint2{ max(a, b.x), max(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 max(const uint2& a, const uint2& b)
{
	return uint2{ max(a.x, b.x), max(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE uint2 clamp(const uint2& a, const unsigned int x, const unsigned int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE uint2 clamp(const uint2& a, const uint2& x, const uint2& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE unsigned int dot(const uint2& a, const uint2& b)
{
	return a.x * b.x + a.y * b.y;
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const int3& a)
{
	return uint3{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y), static_cast<unsigned int>(a.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const float3& a)
{
	return uint3{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y), static_cast<unsigned int>(a.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const unsigned int a)
{
	return uint3{ a, a, a };
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const uint2& a, const unsigned int b)
{
	return uint3{ a.x, a.y, b };
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const unsigned int a, const uint2& b)
{
	return uint3{ a, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE uint3 make_uint3(const uint4& a)
{
	return uint3{ a.x, a.y, a.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const uint3& a, const uint3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const uint3& a, const uint3& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

CU_INLINE CU_HOST_DEVICE void operator++(uint3& a)
{
	a.x++;
	a.y++;
	a.z++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint3& a, const unsigned int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint3& a, const uint3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

CU_INLINE CU_HOST_DEVICE uint3 operator+(const uint3& a, const unsigned int b)
{
	return uint3{ a.x + b, a.y + b, a.z + b };
}

CU_INLINE CU_HOST_DEVICE uint3 operator+(const unsigned int a, const uint3& b)
{
	return uint3{ a + b.x, a + b.y, a + b.z };
}

CU_INLINE CU_HOST_DEVICE uint3 operator+(const uint3& a, const uint3& b)
{
	return uint3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

CU_INLINE CU_HOST_DEVICE void operator--(uint3& a)
{
	a.x--;
	a.y--;
	a.z--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint3& a, const unsigned int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint3& a, const uint3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

CU_INLINE CU_HOST_DEVICE uint3 operator-(const uint3& a, const unsigned int b)
{
	return uint3{ a.x - b, a.y - b, a.z - b };
}

CU_INLINE CU_HOST_DEVICE uint3 operator-(const unsigned int a, const uint3& b)
{
	return uint3{ a - b.x, a - b.y, a - b.z };
}

CU_INLINE CU_HOST_DEVICE uint3 operator-(const uint3& a, const uint3& b)
{
	return uint3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint3& a, const unsigned int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint3& a, const uint3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

CU_INLINE CU_HOST_DEVICE uint3 operator*(const uint3& a, const unsigned int b)
{
	return uint3{ a.x * b, a.y * b, a.z * b };
}

CU_INLINE CU_HOST_DEVICE uint3 operator*(const unsigned int a, const uint3& b)
{
	return uint3{ a * b.x, a * b.y, a * b.z };
}

CU_INLINE CU_HOST_DEVICE uint3 operator*(const uint3& a, const uint3& b)
{
	return uint3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint3& a, const unsigned int b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint3& a, const uint3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}

CU_INLINE CU_HOST_DEVICE uint3 operator/(const uint3& a, const unsigned int b)
{
	return uint3{ a.x / b, a.y / b, a.z / b };
}

CU_INLINE CU_HOST_DEVICE uint3 operator/(const unsigned int a, const uint3& b)
{
	return uint3{ a / b.x, a / b.y, a / b.z };
}

CU_INLINE CU_HOST_DEVICE uint3 operator/(const uint3& a, const uint3& b)
{
	return uint3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

CU_INLINE CU_HOST_DEVICE uint3 min(const uint3& a, const unsigned int b)
{
	return uint3{ min(a.x, b), min(a.y, b), min(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE uint3 min(const unsigned int a, const uint3& b)
{
	return uint3{ min(a, b.x), min(a, b.y), min(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 min(const uint3& a, const uint3& b)
{
	return uint3{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 max(const uint3& a, const unsigned int b)
{
	return uint3{ max(a.x, b), max(a.y, b), max(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE uint3 max(const unsigned int a, const uint3& b)
{
	return uint3{ max(a, b.x), max(a, b.y), max(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 max(const uint3& a, const uint3& b)
{
	return uint3{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE uint3 clamp(const uint3& a, const unsigned int x, const unsigned int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE uint3 clamp(const uint3& a, const uint3& x, const uint3& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE unsigned int dot(const uint3& a, const uint3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const int4& a)
{
	return uint4{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y), static_cast<unsigned int>(a.z), static_cast<unsigned int>(a.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const float4& a)
{
	return uint4{ static_cast<unsigned int>(a.x), static_cast<unsigned int>(a.y), static_cast<unsigned int>(a.z), static_cast<unsigned int>(a.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const unsigned int a)
{
	return uint4{ a, a, a, a };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const uint2& a, const unsigned int b, const unsigned int c)
{
	return uint4{ a.x, a.y, b, c };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const unsigned int a, const uint2& b, const unsigned int c)
{
	return uint4{ a, b.x, b.y, c };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const unsigned int a, const unsigned int b, const uint2& c)
{
	return uint4{ a, b, c.x, c.y };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const uint2& a, const uint2& b)
{
	return uint4{ a.x, a.y, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const uint3& a, const unsigned int b)
{
	return uint4{ a.x, a.y, a.z, b };
}

CU_INLINE CU_HOST_DEVICE uint4 make_uint4(const unsigned int a, const uint3& b)
{
	return uint4{ a, b.x, b.y, b.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const uint4& a, const uint4& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const uint4& a, const uint4& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

CU_INLINE CU_HOST_DEVICE void operator++(uint4& a)
{
	a.x++;
	a.y++;
	a.z++;
	a.w++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint4& a, const unsigned int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(uint4& a, const uint4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

CU_INLINE CU_HOST_DEVICE uint4 operator+(const uint4& a, const unsigned int b)
{
	return uint4{ a.x + b, a.y + b, a.z + b, a.w + b };
}

CU_INLINE CU_HOST_DEVICE uint4 operator+(const unsigned int a, const uint4& b)
{
	return uint4{ a + b.x, a + b.y, a + b.z, a + b.w };
}

CU_INLINE CU_HOST_DEVICE uint4 operator+(const uint4& a, const uint4& b)
{
	return uint4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

CU_INLINE CU_HOST_DEVICE void operator--(uint4& a)
{
	a.x--;
	a.y--;
	a.z--;
	a.w--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint4& a, const unsigned int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(uint4& a, const uint4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

CU_INLINE CU_HOST_DEVICE uint4 operator-(const uint4& a, const unsigned int b)
{
	return uint4{ a.x - b, a.y - b, a.z - b, a.w - b };
}

CU_INLINE CU_HOST_DEVICE uint4 operator-(const unsigned int a, const uint4& b)
{
	return uint4{ a - b.x, a - b.y, a - b.z, a - b.w };
}

CU_INLINE CU_HOST_DEVICE uint4 operator-(const uint4& a, const uint4& b)
{
	return uint4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint4& a, const unsigned int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(uint4& a, const uint4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

CU_INLINE CU_HOST_DEVICE uint4 operator*(const uint4& a, const unsigned int b)
{
	return uint4{ a.x * b, a.y * b, a.z * b, a.w * b };
}

CU_INLINE CU_HOST_DEVICE uint4 operator*(const unsigned int a, const uint4& b)
{
	return uint4{ a * b.x, a * b.y, a * b.z, a * b.w };
}

CU_INLINE CU_HOST_DEVICE uint4 operator*(const uint4& a, const uint4& b)
{
	return uint4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint4& a, const unsigned int b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(uint4& a, const uint4& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

CU_INLINE CU_HOST_DEVICE uint4 operator/(const uint4& a, const unsigned int b)
{
	return uint4{ a.x / b, a.y / b, a.z / b, a.w / b };
}

CU_INLINE CU_HOST_DEVICE uint4 operator/(const unsigned int a, const uint4& b)
{
	return uint4{ a / b.x, a / b.y, a / b.z, a / b.w };
}

CU_INLINE CU_HOST_DEVICE uint4 operator/(const uint4& a, const uint4& b)
{
	return uint4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

CU_INLINE CU_HOST_DEVICE uint4 min(const uint4& a, const unsigned int b)
{
	return uint4{ min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE uint4 min(const unsigned int a, const uint4& b)
{
	return uint4{ min(a, b.x), min(a, b.y), min(a, b.z), min(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 min(const uint4& a, const uint4& b)
{
	return uint4{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 max(const uint4& a, const unsigned int b)
{
	return uint4{ max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE uint4 max(const unsigned int a, const uint4& b)
{
	return uint4{ max(a, b.x), max(a, b.y), max(a, b.z), max(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 max(const uint4& a, const uint4& b)
{
	return uint4{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE uint4 clamp(const uint4& a, const unsigned int x, const unsigned int y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE uint4 clamp(const uint4& a, const uint4& x, const uint4& y)
{
	return max(x, min(a, y));
}

CU_INLINE CU_HOST_DEVICE unsigned int dot(const uint4& a, const uint4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

CU_INLINE CU_HOST_DEVICE float2 make_float2(const int2& a)
{
	return float2{ static_cast<float>(a.x), static_cast<float>(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 make_float2(const uint2& a)
{
	return float2{ static_cast<float>(a.x), static_cast<float>(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 make_float2(const float a)
{
	return float2{ a, a };
}

CU_INLINE CU_HOST_DEVICE float2 make_float2(const float3& a)
{
	return float2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE float2 make_float2(const float4& a)
{
	return float2{ a.x, a.y };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const float2& a, const float2& b)
{
	return a.x == b.x && a.y == b.y;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const float2& a, const float2& b)
{
	return a.x != b.x || a.y != b.y;
}

CU_INLINE CU_HOST_DEVICE void operator++(float2& a)
{
	a.x++;
	a.y++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float2& a, const float b)
{
	a.x += b;
	a.y += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float2& a, const float2& b)
{
	a.x += b.x;
	a.y += b.y;
}

CU_INLINE CU_HOST_DEVICE float2 operator+(const float2& a, const float b)
{
	return float2{ a.x + b, a.y + b };
}

CU_INLINE CU_HOST_DEVICE float2 operator+(const float a, const float2& b)
{
	return float2{ a + b.x, a + b.y };
}

CU_INLINE CU_HOST_DEVICE float2 operator+(const float2& a, const float2& b)
{
	return float2{ a.x + b.x, a.y + b.y };
}

CU_INLINE CU_HOST_DEVICE void operator--(float2& a)
{
	a.x--;
	a.y--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float2& a, const float b)
{
	a.x -= b;
	a.y -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float2& a, const float2& b)
{
	a.x -= b.x;
	a.y -= b.y;
}

CU_INLINE CU_HOST_DEVICE float2 operator-(const float2& a)
{
	return float2{ -a.x, -a.y };
}

CU_INLINE CU_HOST_DEVICE float2 operator-(const float2& a, const float b)
{
	return float2{ a.x - b, a.y - b };
}

CU_INLINE CU_HOST_DEVICE float2 operator-(const float a, const float2& b)
{
	return float2{ a - b.x, a - b.y };
}

CU_INLINE CU_HOST_DEVICE float2 operator-(const float2& a, const float2& b)
{
	return float2{ a.x - b.x, a.y - b.y };
}

CU_INLINE CU_HOST_DEVICE void operator*=(float2& a, const float b)
{
	a.x *= b;
	a.y *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(float2& a, const float2& b)
{
	a.x *= b.x;
	a.y *= b.y;
}

CU_INLINE CU_HOST_DEVICE float2 operator*(const float2& a, const float b)
{
	return float2{ a.x * b, a.y * b };
}

CU_INLINE CU_HOST_DEVICE float2 operator*(const float a, const float2& b)
{
	return float2{ a * b.x, a * b.y };
}

CU_INLINE CU_HOST_DEVICE float2 operator*(const float2& a, const float2& b)
{
	return float2{ a.x * b.x, a.y * b.y };
}

CU_INLINE CU_HOST_DEVICE void operator/=(float2& a, const float b)
{
	a.x /= b;
	a.y /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(float2& a, const float2& b)
{
	a.x /= b.x;
	a.y /= b.y;
}

CU_INLINE CU_HOST_DEVICE float2 operator/(const float2& a, const float b)
{
	return float2{ a.x / b, a.y / b };
}

CU_INLINE CU_HOST_DEVICE float2 operator/(const float a, const float2& b)
{
	return float2{ a / b.x, a / b.y };
}

CU_INLINE CU_HOST_DEVICE float2 operator/(const float2& a, const float2& b)
{
	return float2{ a.x / b.x, a.y / b.y };
}

CU_INLINE CU_HOST_DEVICE float2 fabsf(const float2& a)
{
	return float2{ fabsf(a.x), fabsf(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 fminf(const float2& a, const float b)
{
	return float2{ fminf(a.x, b), fminf(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE float2 fminf(const float a, const float2& b)
{
	return float2{ fminf(a, b.x), fminf(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE float2 fminf(const float2& a, const float2& b)
{
	return float2{ fminf(a.x, b.x), fminf(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE float2 fmaxf(const float2& a, const float b)
{
	return float2{ fmaxf(a.x, b), fmaxf(a.y, b) };
}

CU_INLINE CU_HOST_DEVICE float2 fmaxf(const float a, const float2& b)
{
	return float2{ fmaxf(a, b.x), fmaxf(a, b.y) };
}

CU_INLINE CU_HOST_DEVICE float2 fmaxf(const float2& a, const float2& b)
{
	return float2{ fmaxf(a.x, b.x), fmaxf(a.y, b.y) };
}

CU_INLINE CU_HOST_DEVICE float2 floorf(const float2& a)
{
	return float2{ floorf(a.x), floorf(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 ceilf(const float2& a)
{
	return float2{ ceilf(a.x), ceilf(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 roundf(const float2& a)
{
	return float2{ roundf(a.x), roundf(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 sign(const float2& a)
{
	return float2{ sign(a.x), sign(a.y) };
}

CU_INLINE CU_HOST_DEVICE float2 clamp(const float2& a, const float x, const float y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float2 clamp(const float2& a, const float2& x, const float2& y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

CU_INLINE CU_HOST_DEVICE float length(const float2& a)
{
	return sqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float2 normalize(const float2& a)
{
	return a * rsqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float2 reflect(const float2& a, const float2& n)
{
	return a - 2.0f * n * dot(a, n);
}

CU_INLINE CU_HOST_DEVICE float2 lerp(const float2 a, const float2 b, const float t)
{
	return a + t * (b - a);
}

CU_INLINE CU_HOST_DEVICE float2 bilerp(const float2 a00, const float2 a10, const float2 a01, const float2 a11, const float u, const float v)
{
	return lerp(lerp(a00, a10, u), lerp(a01, a11, u), v);
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const int3& a)
{
	return float3{ static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const uint3& a)
{
	return float3{ static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const float a)
{
	return float3{ a, a, a };
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const float2& a, const float b)
{
	return float3{ a.x, a.y, b };
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const float a, const float2& b)
{
	return float3{ a, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE float3 make_float3(const float4& a)
{
	return float3{ a.x, a.y, a.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const float3& a, const float3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const float3& a, const float3& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

CU_INLINE CU_HOST_DEVICE void operator++(float3& a)
{
	a.x++;
	a.y++;
	a.z++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float3& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

CU_INLINE CU_HOST_DEVICE float3 operator+(const float3& a, const float b)
{
	return float3{ a.x + b, a.y + b, a.z + b };
}

CU_INLINE CU_HOST_DEVICE float3 operator+(const float a, const float3& b)
{
	return float3{ a + b.x, a + b.y, a + b.z };
}

CU_INLINE CU_HOST_DEVICE float3 operator+(const float3& a, const float3& b)
{
	return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

CU_INLINE CU_HOST_DEVICE void operator--(float3& a)
{
	a.x--;
	a.y--;
	a.z--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float3& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

CU_INLINE CU_HOST_DEVICE float3 operator-(const float3& a)
{
	return float3{ -a.x, -a.y, -a.z };
}

CU_INLINE CU_HOST_DEVICE float3 operator-(const float3& a, const float b)
{
	return float3{ a.x - b, a.y - b, a.z - b };
}

CU_INLINE CU_HOST_DEVICE float3 operator-(const float a, const float3& b)
{
	return float3{ a - b.x, a - b.y, a - b.z };
}

CU_INLINE CU_HOST_DEVICE float3 operator-(const float3& a, const float3& b)
{
	return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

CU_INLINE CU_HOST_DEVICE void operator*=(float3& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(float3& a, const float3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

CU_INLINE CU_HOST_DEVICE float3 operator*(const float3& a, const float b)
{
	return float3{ a.x * b, a.y * b, a.z * b };
}

CU_INLINE CU_HOST_DEVICE float3 operator*(const float a, const float3& b)
{
	return float3{ a * b.x, a * b.y, a * b.z };
}

CU_INLINE CU_HOST_DEVICE float3 operator*(const float3& a, const float3& b)
{
	return float3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

CU_INLINE CU_HOST_DEVICE void operator/=(float3& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(float3& a, const float3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}

CU_INLINE CU_HOST_DEVICE float3 operator/(const float3& a, const float b)
{
	return float3{ a.x / b, a.y / b, a.z / b };
}

CU_INLINE CU_HOST_DEVICE float3 operator/(const float a, const float3& b)
{
	return float3{ a / b.x, a / b.y, a / b.z };
}

CU_INLINE CU_HOST_DEVICE float3 operator/(const float3& a, const float3& b)
{
	return float3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

CU_INLINE CU_HOST_DEVICE float3 fabsf(const float3& a)
{
	return float3{ fabsf(a.x), fabsf(a.y), fabsf(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 fminf(const float3& a, const float b)
{
	return float3{ fminf(a.x, b), fminf(a.y, b), fminf(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE float3 fminf(const float a, const float3& b)
{
	return float3{ fminf(a, b.x), fminf(a, b.y), fminf(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE float3 fminf(const float3& a, const float3& b)
{
	return float3{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE float3 fmaxf(const float3& a, const float b)
{
	return float3{ fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b) };
}

CU_INLINE CU_HOST_DEVICE float3 fmaxf(const float a, const float3& b)
{
	return float3{ fmaxf(a, b.x), fmaxf(a, b.y), fmaxf(a, b.z) };
}

CU_INLINE CU_HOST_DEVICE float3 fmaxf(const float3& a, const float3& b)
{
	return float3{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) };
}

CU_INLINE CU_HOST_DEVICE float3 floorf(const float3& a)
{
	return float3{ floorf(a.x), floorf(a.y), floorf(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 ceilf(const float3& a)
{
	return float3{ ceilf(a.x), ceilf(a.y), ceilf(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 roundf(const float3& a)
{
	return float3{ roundf(a.x), roundf(a.y), roundf(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 sign(const float3& a)
{
	return float3{ sign(a.x), sign(a.y), sign(a.z) };
}

CU_INLINE CU_HOST_DEVICE float3 clamp(const float3& a, const float x, const float y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float3 clamp(const float3& a, const float3& x, const float3& y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

CU_INLINE CU_HOST_DEVICE float length(const float3& a)
{
	return sqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float3 normalize(const float3& a)
{
	return a * rsqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float3 cross(const float3& a, const float3& b)
{
	return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

CU_INLINE CU_HOST_DEVICE float3 reflect(const float3& a, const float3& n)
{
	return a - 2.0f * n * dot(a, n);
}

CU_INLINE CU_HOST_DEVICE float3 lerp(const float3 a, const float3 b, const float t)
{
	return a + t * (b - a);
}

CU_INLINE CU_HOST_DEVICE float3 bilerp(const float3 a00, const float3 a10, const float3 a01, const float3 a11, const float u, const float v)
{
	return lerp(lerp(a00, a10, u), lerp(a01, a11, u), v);
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const int4& a)
{
	return float4{ static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const uint4& a)
{
	return float4{ static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z), static_cast<float>(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float a)
{
	return float4{ a, a, a, a };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float2& a, const float b, const float c)
{
	return float4{ a.x, a.y, b, c };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float a, const float2& b, const float c)
{
	return float4{ a, b.x, b.y, c };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float a, const float b, const float2& c)
{
	return float4{ a, b, c.x, c.y };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float2& a, const float2& b)
{
	return float4{ a.x, a.y, b.x, b.y };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float3& a, const float b)
{
	return float4{ a.x, a.y, a.z, b };
}

CU_INLINE CU_HOST_DEVICE float4 make_float4(const float a, const float3& b)
{
	return float4{ a, b.x, b.y, b.z };
}

CU_INLINE CU_HOST_DEVICE bool operator==(const float4& a, const float4& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CU_INLINE CU_HOST_DEVICE bool operator!=(const float4& a, const float4& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

CU_INLINE CU_HOST_DEVICE void operator++(float4& a)
{
	a.x++;
	a.y++;
	a.z++;
	a.w++;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float4& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

CU_INLINE CU_HOST_DEVICE void operator+=(float4& a, const float4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

CU_INLINE CU_HOST_DEVICE float4 operator+(const float4& a, const float b)
{
	return float4{ a.x + b, a.y + b, a.z + b, a.w + b };
}

CU_INLINE CU_HOST_DEVICE float4 operator+(const float a, const float4& b)
{
	return float4{ a + b.x, a + b.y, a + b.z, a + b.w };
}

CU_INLINE CU_HOST_DEVICE float4 operator+(const float4& a, const float4& b)
{
	return float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

CU_INLINE CU_HOST_DEVICE void operator--(float4& a)
{
	a.x--;
	a.y--;
	a.z--;
	a.w--;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float4& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

CU_INLINE CU_HOST_DEVICE void operator-=(float4& a, const float4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

CU_INLINE CU_HOST_DEVICE float4 operator-(const float4& a)
{
	return float4{ -a.x, -a.y, -a.z, -a.w };
}

CU_INLINE CU_HOST_DEVICE float4 operator-(const float4& a, const float b)
{
	return float4{ a.x - b, a.y - b, a.z - b, a.w - b };
}

CU_INLINE CU_HOST_DEVICE float4 operator-(const float a, const float4& b)
{
	return float4{ a - b.x, a - b.y, a - b.z, a - b.w };
}

CU_INLINE CU_HOST_DEVICE float4 operator-(const float4& a, const float4& b)
{
	return float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

CU_INLINE CU_HOST_DEVICE void operator*=(float4& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

CU_INLINE CU_HOST_DEVICE void operator*=(float4& a, const float4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

CU_INLINE CU_HOST_DEVICE float4 operator*(const float4& a, const float b)
{
	return float4{ a.x * b, a.y * b, a.z * b, a.w * b };
}

CU_INLINE CU_HOST_DEVICE float4 operator*(const float a, const float4& b)
{
	return float4{ a * b.x, a * b.y, a * b.z, a * b.w };
}

CU_INLINE CU_HOST_DEVICE float4 operator*(const float4& a, const float4& b)
{
	return float4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

CU_INLINE CU_HOST_DEVICE void operator/=(float4& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}

CU_INLINE CU_HOST_DEVICE void operator/=(float4& a, const float4& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

CU_INLINE CU_HOST_DEVICE float4 operator/(const float4& a, const float b)
{
	return float4{ a.x / b, a.y / b, a.z / b, a.w / b };
}

CU_INLINE CU_HOST_DEVICE float4 operator/(const float a, const float4& b)
{
	return float4{ a / b.x, a / b.y, a / b.z, a / b.w };
}

CU_INLINE CU_HOST_DEVICE float4 operator/(const float4& a, const float4& b)
{
	return float4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

CU_INLINE CU_HOST_DEVICE float4 fabsf(const float4& a)
{
	return float4{ fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 fminf(const float4& a, const float b)
{
	return float4{ fminf(a.x, b), fminf(a.y, b), fminf(a.z, b), fminf(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE float4 fminf(const float a, const float4& b)
{
	return float4{ fminf(a, b.x), fminf(a, b.y), fminf(a, b.z), fminf(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE float4 fminf(const float4& a, const float4& b)
{
	return float4{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE float4 fmaxf(const float4& a, const float b)
{
	return float4{ fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b), fmaxf(a.w, b) };
}

CU_INLINE CU_HOST_DEVICE float4 fmaxf(const float a, const float4& b)
{
	return float4{ fmaxf(a, b.x), fmaxf(a, b.y), fmaxf(a, b.z), fmaxf(a, b.w) };
}

CU_INLINE CU_HOST_DEVICE float4 fmaxf(const float4& a, const float4& b)
{
	return float4{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w) };
}

CU_INLINE CU_HOST_DEVICE float4 floorf(const float4& a)
{
	return float4{ floorf(a.x), floorf(a.y), floorf(a.z), floor(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 ceilf(const float4& a)
{
	return float4{ ceilf(a.x), ceilf(a.y), ceilf(a.z), ceilf(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 roundf(const float4& a)
{
	return float4{ roundf(a.x), roundf(a.y), roundf(a.z), roundf(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 sign(const float4& a)
{
	return float4{ sign(a.x), sign(a.y), sign(a.z), sign(a.w) };
}

CU_INLINE CU_HOST_DEVICE float4 clamp(const float4& a, const float x, const float y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float4 clamp(const float4& a, const float4& x, const float4& y)
{
	return fmaxf(x, fminf(a, y));
}

CU_INLINE CU_HOST_DEVICE float dot(const float4& a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

CU_INLINE CU_HOST_DEVICE float length(const float4& a)
{
	return sqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float4 normalize(const float4& a)
{
	return a * rsqrtf(dot(a, a));
}

CU_INLINE CU_HOST_DEVICE float4 reflect(const float4& a, const float4& n)
{
	return a - 2.0f * n * dot(a, n);
}

CU_INLINE CU_HOST_DEVICE float4 lerp(const float4 a, const float4 b, const float t)
{
	return a + t * (b - a);
}

CU_INLINE CU_HOST_DEVICE float4 bilerp(const float4 a00, const float4 a10, const float4 a01, const float4 a11, const float u, const float v)
{
	return lerp(lerp(a00, a10, u), lerp(a01, a11, u), v);
}
