#pragma once

/**
 * ScanContext utility functions extracted from sc_pgo/Scancontext.h/.cpp
 * Pure algorithm functions — no SCManager, no GTSAM dependency.
 */

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace sc_utils {

// Hyperparameters (matching sc_pgo defaults)
constexpr int PC_NUM_RING = 20;
constexpr int PC_NUM_SECTOR = 60;
constexpr double LIDAR_HEIGHT = 2.0;

inline float xy2theta(float x, float y) {
  if (x >= 0 && y >= 0) return (180.0f / M_PI) * std::atan(y / x);
  if (x < 0  && y >= 0) return 180.0f - (180.0f / M_PI) * std::atan(y / (-x));
  if (x < 0  && y < 0)  return 180.0f + (180.0f / M_PI) * std::atan(y / x);
  return 360.0f - (180.0f / M_PI) * std::atan((-y) / x);
}

inline Eigen::MatrixXd circshift(const Eigen::MatrixXd& mat, int num_shift) {
  if (num_shift == 0) return mat;
  Eigen::MatrixXd shifted = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
  for (int c = 0; c < mat.cols(); c++) {
    shifted.col((c + num_shift) % mat.cols()) = mat.col(c);
  }
  return shifted;
}

inline std::vector<float> eig2stdvec(const Eigen::MatrixXd& m) {
  return std::vector<float>(m.data(), m.data() + m.size());
}

/**
 * Create ScanContext descriptor (PC_NUM_RING x PC_NUM_SECTOR) from point cloud
 */
inline Eigen::MatrixXd makeScancontext(const pcl::PointCloud<pcl::PointXYZI>& scan, double max_radius) {
  const int NO_POINT = -1000;
  Eigen::MatrixXd desc = NO_POINT * Eigen::MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

  const double unit_ring_gap = max_radius / PC_NUM_RING;
  const double unit_sector_angle = 360.0 / PC_NUM_SECTOR;

  for (const auto& pt_raw : scan.points) {
    float x = pt_raw.x;
    float y = pt_raw.y;
    float z = pt_raw.z + LIDAR_HEIGHT;

    float range = std::sqrt(x * x + y * y);
    if (range > max_radius) continue;

    float angle = xy2theta(x, y);
    int ring_idx = std::max(std::min(PC_NUM_RING, (int)std::ceil((range / max_radius) * PC_NUM_RING)), 1);
    int sector_idx = std::max(std::min(PC_NUM_SECTOR, (int)std::ceil((angle / 360.0) * PC_NUM_SECTOR)), 1);

    if (desc(ring_idx - 1, sector_idx - 1) < z) {
      desc(ring_idx - 1, sector_idx - 1) = z;
    }
  }

  for (int r = 0; r < desc.rows(); r++)
    for (int c = 0; c < desc.cols(); c++)
      if (desc(r, c) == NO_POINT) desc(r, c) = 0;

  return desc;
}

/**
 * Overload for PointXYZ (convert to XYZI internally)
 */
inline Eigen::MatrixXd makeScancontext(const pcl::PointCloud<pcl::PointXYZ>& scan, double max_radius) {
  const int NO_POINT = -1000;
  Eigen::MatrixXd desc = NO_POINT * Eigen::MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

  for (const auto& pt_raw : scan.points) {
    float x = pt_raw.x;
    float y = pt_raw.y;
    float z = pt_raw.z + LIDAR_HEIGHT;

    float range = std::sqrt(x * x + y * y);
    if (range > max_radius) continue;

    float angle = xy2theta(x, y);
    int ring_idx = std::max(std::min(PC_NUM_RING, (int)std::ceil((range / max_radius) * PC_NUM_RING)), 1);
    int sector_idx = std::max(std::min(PC_NUM_SECTOR, (int)std::ceil((angle / 360.0) * PC_NUM_SECTOR)), 1);

    if (desc(ring_idx - 1, sector_idx - 1) < z) {
      desc(ring_idx - 1, sector_idx - 1) = z;
    }
  }

  for (int r = 0; r < desc.rows(); r++)
    for (int c = 0; c < desc.cols(); c++)
      if (desc(r, c) == NO_POINT) desc(r, c) = 0;

  return desc;
}

inline Eigen::MatrixXd makeRingkeyFromScancontext(const Eigen::MatrixXd& desc) {
  Eigen::MatrixXd key(desc.rows(), 1);
  for (int r = 0; r < desc.rows(); r++) {
    key(r, 0) = desc.row(r).mean();
  }
  return key;
}

inline Eigen::MatrixXd makeSectorkeyFromScancontext(const Eigen::MatrixXd& desc) {
  Eigen::MatrixXd key(1, desc.cols());
  for (int c = 0; c < desc.cols(); c++) {
    key(0, c) = desc.col(c).mean();
  }
  return key;
}

inline double distDirectSC(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
  int num_eff_cols = 0;
  double sum_similarity = 0;
  for (int c = 0; c < sc1.cols(); c++) {
    Eigen::VectorXd c1 = sc1.col(c);
    Eigen::VectorXd c2 = sc2.col(c);
    if (c1.norm() == 0 || c2.norm() == 0) continue;
    sum_similarity += c1.dot(c2) / (c1.norm() * c2.norm());
    num_eff_cols++;
  }
  if (num_eff_cols == 0) return 1.0;
  return 1.0 - sum_similarity / num_eff_cols;
}

inline int fastAlignUsingVkey(const Eigen::MatrixXd& vkey1, const Eigen::MatrixXd& vkey2) {
  int best_shift = 0;
  double min_diff = 1e10;
  for (int s = 0; s < vkey1.cols(); s++) {
    Eigen::MatrixXd shifted = circshift(vkey2, s);
    double diff = (vkey1 - shifted).norm();
    if (diff < min_diff) {
      min_diff = diff;
      best_shift = s;
    }
  }
  return best_shift;
}

/**
 * @return (distance, best_shift_columns)
 */
inline std::pair<double, int> distanceBtnScanContext(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
  Eigen::MatrixXd vkey1 = makeSectorkeyFromScancontext(sc1);
  Eigen::MatrixXd vkey2 = makeSectorkeyFromScancontext(sc2);
  int init_shift = fastAlignUsingVkey(vkey1, vkey2);

  const double SEARCH_RATIO = 0.1;
  int search_radius = std::round(0.5 * SEARCH_RATIO * sc1.cols());
  std::vector<int> shifts{init_shift};
  for (int i = 1; i <= search_radius; i++) {
    shifts.push_back((init_shift + i + (int)sc1.cols()) % sc1.cols());
    shifts.push_back((init_shift - i + (int)sc1.cols()) % sc1.cols());
  }

  int best_shift = 0;
  double min_dist = 1e10;
  for (int s : shifts) {
    Eigen::MatrixXd sc2_shifted = circshift(sc2, s);
    double d = distDirectSC(sc1, sc2_shifted);
    if (d < min_dist) {
      min_dist = d;
      best_shift = s;
    }
  }
  return {min_dist, best_shift};
}

}  // namespace sc_utils
