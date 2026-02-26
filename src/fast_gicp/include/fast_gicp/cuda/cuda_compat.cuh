// CUDA 12.x Compatibility Header for fast_gicp
// Resolves thrust/libcu++ namespace conflicts

#ifndef FAST_GICP_CUDA_COMPAT_CUH
#define FAST_GICP_CUDA_COMPAT_CUH

#include <cuda_runtime.h>
#include <Eigen/Core>

// CUDA 12.x has integrated thrust with libcu++, causing namespace conflicts
// We define our own simple pair to avoid thrust::pair ambiguity

namespace fast_gicp {
namespace cuda {

// Simple pair replacement to avoid thrust::pair ambiguity in CUDA 12.x
template<typename T1, typename T2>
struct Pair {
    T1 first;
    T2 second;

    __host__ __device__ Pair() : first(), second() {}
    __host__ __device__ Pair(const T1& a, const T2& b) : first(a), second(b) {}

    __host__ __device__ bool operator==(const Pair& other) const {
        return first == other.first && second == other.second;
    }

    __host__ __device__ bool operator!=(const Pair& other) const {
        return !(*this == other);
    }

    __host__ __device__ bool operator<(const Pair& other) const {
        if (first < other.first) return true;
        if (other.first < first) return false;
        return second < other.second;
    }
};

template<typename T1, typename T2>
__host__ __device__ Pair<T1, T2> make_pair(const T1& a, const T2& b) {
    return Pair<T1, T2>(a, b);
}

// Common pair types using our custom pair
using IntPair = Pair<int, int>;
using FloatIntPair = Pair<float, int>;
using VoxelBucket = Pair<Eigen::Vector3i, int>;  // For gaussian voxelmap

// Functor for comparing Pairs in thrust algorithms
template<typename T1, typename T2>
struct PairLess {
    __host__ __device__ bool operator()(
        const Pair<T1, T2>& lhs,
        const Pair<T1, T2>& rhs) const {
        return lhs < rhs;
    }
};

using VoxelBucketLess = PairLess<Eigen::Vector3i, int>;

}  // namespace cuda
}  // namespace fast_gicp

#endif  // FAST_GICP_CUDA_COMPAT_CUH
