#pragma once

#if !defined(__CUDACC__) && !defined(__CUDABE__)
#    define CU_INLINE inline
#    define CU_HOST_DEVICE
#else
#    define CU_INLINE __forceinline__
#    define CU_HOST_DEVICE __host__ __device__
#endif
