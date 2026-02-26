#ifndef FAST_GICP_CUDA_BRUTE_FORCE_KNN_CUH
#define FAST_GICP_CUDA_BRUTE_FORCE_KNN_CUH

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <fast_gicp/cuda/cuda_compat.cuh>

namespace fast_gicp {
namespace cuda {

// Use custom FloatIntPair instead of thrust::pair<float, int>
void brute_force_knn_search(
    const thrust::device_vector<Eigen::Vector3f>& source,
    const thrust::device_vector<Eigen::Vector3f>& target,
    int k,
    thrust::device_vector<FloatIntPair>& k_neighbors,
    bool do_sort = false);

}  // namespace cuda
}  // namespace fast_gicp

#endif
