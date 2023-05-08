#pragma once

#ifndef __CUDACC_RTC__
#    define CU_INLINE inline
#    define CU_HOST_DEVICE
#else
#    define CU_INLINE __forceinline__
#    define CU_HOST_DEVICE __host__ __device__
#endif
