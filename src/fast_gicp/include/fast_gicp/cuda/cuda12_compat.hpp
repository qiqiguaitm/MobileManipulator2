// CUDA 12.x compatibility header for fast_gicp
// Fixes thrust namespace conflicts with libcu++

#ifndef FAST_GICP_CUDA12_COMPAT_HPP
#define FAST_GICP_CUDA12_COMPAT_HPP

// Check CUDA version
#if defined(__CUDACC__)
#include <cuda_runtime.h>

#if CUDART_VERSION >= 12000
// CUDA 12.x specific fixes

// Disable thrust/libcu++ interoperability that causes pair ambiguity
#ifndef _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#define _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#endif

// Use thrust pair explicitly
#define THRUST_PAIR ::thrust::pair

#else
// CUDA 11.x and earlier
#define THRUST_PAIR thrust::pair

#endif // CUDART_VERSION >= 12000

#else
// Non-CUDA compilation
#define THRUST_PAIR thrust::pair

#endif // __CUDACC__

#endif // FAST_GICP_CUDA12_COMPAT_HPP
