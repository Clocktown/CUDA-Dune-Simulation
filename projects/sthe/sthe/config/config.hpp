#pragma once

#include <cassert>

static_assert(sizeof(bool) == 1, "Bool must have a size of 1 byte");
static_assert(sizeof(short) == 2, "Short must have a size of 2 bytes");
static_assert(sizeof(int) == 4, "Int must have a size of 4 bytes");
static_assert(sizeof(unsigned int) == 4, "Unsigned int must have a size of 4 bytes");
static_assert(sizeof(long long int) == 8, "Long long int must have a size of 8 bytes");
static_assert(sizeof(unsigned long long int) == 8, "Unsigned long long int must have a size of 8 bytes");
