/**
 * build_sc_database: 构建 ScanContext 数据库
 *
 * 两种模式：
 *   1. 真实扫描模式 (推荐): 从 Scans/*.pcd + optimized_poses.txt 构建
 *   2. 虚拟视点模式 (fallback): 从 GlobalMap_hdl.pcd 均匀采样虚拟视点
 *
 * Usage:
 *   # 模式1: 真实扫描 (建图后使用)
 *   ros2 run hdl_global_localization build_sc_database \
 *     --ros-args -p scan_dir:=/path/to/maps/sc_pgo -p output:=sc_database.bin \
 *     -p radius:=4.0
 *
 *   # 模式2: 虚拟视点 (无真实扫描时)
 *   ros2 run hdl_global_localization build_sc_database \
 *     --ros-args -p input:=GlobalMap_hdl.pcd -p output:=sc_database.bin \
 *     -p grid_step:=1.0 -p radius:=4.0
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <hdl_global_localization/sc/sc_utils.hpp>

// Binary database format (与 global_localization_sc.cpp 一致)
struct SCDatabaseHeader {
  char magic[8];       // "SCDB_V01"
  uint32_t count;      // number of entries
  uint32_t num_ring;   // PC_NUM_RING
  uint32_t num_sector; // PC_NUM_SECTOR
  double max_radius;   // PC_MAX_RADIUS used
  char reserved[32];   // future use
};

struct Entry {
  Eigen::Matrix4d pose;
  Eigen::MatrixXd desc;        // 20x60
  std::vector<float> ring_key; // 20
};

class BuildSCDatabaseNode : public rclcpp::Node {
public:
  BuildSCDatabaseNode() : Node("build_sc_database") {
    declare_parameter("input", "");       // 虚拟视点模式: GlobalMap PCD
    declare_parameter("scan_dir", "");    // 真实扫描模式: 包含 Scans/ 和 optimized_poses.txt 的目录
    declare_parameter("output", "");
    declare_parameter("grid_step", 1.0);
    declare_parameter("radius", 4.0);
    declare_parameter("min_density_radius", 2.0);
    declare_parameter("min_points_in_view", 30);

    run();
    rclcpp::shutdown();
  }

private:
  void run() {
    std::string input = get_parameter("input").as_string();
    std::string scan_dir = get_parameter("scan_dir").as_string();
    std::string output = get_parameter("output").as_string();
    double radius = get_parameter("radius").as_double();

    if (output.empty()) {
      RCLCPP_ERROR(get_logger(), "Usage: --ros-args -p output:=<bin> [-p scan_dir:=<dir> | -p input:=<pcd>]");
      return;
    }

    std::vector<Entry> entries;

    if (!scan_dir.empty()) {
      entries = build_from_real_scans(scan_dir, radius);
    } else if (!input.empty()) {
      entries = build_from_virtual_viewpoints(input, radius);
    } else {
      RCLCPP_ERROR(get_logger(), "Must specify either scan_dir (real scans) or input (virtual viewpoints)");
      return;
    }

    if (entries.empty()) {
      RCLCPP_ERROR(get_logger(), "No entries generated!");
      return;
    }

    write_database(output, entries, radius);
  }

  /**
   * 模式1: 从真实扫描 + 优化位姿构建 SC 数据库
   * 输入: scan_dir/Scans/000000.pcd ~ NNNNNN.pcd + scan_dir/optimized_poses.txt
   */
  std::vector<Entry> build_from_real_scans(const std::string& scan_dir, double radius) {
    RCLCPP_INFO(get_logger(), "=== Real Scan Mode ===");

    // 读取 optimized_poses.txt (KITTI format: 12列, 3x4 row-major)
    std::string poses_path = scan_dir + "/optimized_poses.txt";
    std::ifstream poses_file(poses_path);
    if (!poses_file) {
      RCLCPP_ERROR(get_logger(), "Cannot open poses file: %s", poses_path.c_str());
      return {};
    }

    std::vector<Eigen::Matrix4d> poses;
    std::string line;
    while (std::getline(poses_file, line)) {
      if (line.empty()) continue;
      std::istringstream iss(line);
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
          iss >> pose(r, c);
      poses.push_back(pose);
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu poses from %s", poses.size(), poses_path.c_str());

    if (poses.empty()) {
      RCLCPP_ERROR(get_logger(), "No poses loaded!");
      return {};
    }

    // 扫描 Scans/ 目录
    std::string scans_dir = scan_dir + "/Scans";
    std::vector<Entry> entries;

    for (size_t i = 0; i < poses.size(); i++) {
      std::ostringstream ss;
      ss << scans_dir << "/" << std::setfill('0') << std::setw(6) << i << ".pcd";
      std::string scan_path = ss.str();

      pcl::PointCloud<pcl::PointXYZ>::Ptr scan(new pcl::PointCloud<pcl::PointXYZ>());
      if (pcl::io::loadPCDFile(scan_path, *scan) < 0) {
        RCLCPP_WARN(get_logger(), "Skip missing scan: %s", scan_path.c_str());
        continue;
      }

      if (scan->size() < 50) {
        RCLCPP_WARN(get_logger(), "Skip sparse scan %zu (%zu points)", i, scan->size());
        continue;
      }

      // 扫描已经在 body frame (以传感器为中心)，直接计算 SC 描述符
      Eigen::MatrixXd desc = sc_utils::makeScancontext(*scan, radius);
      Eigen::MatrixXd ring_key_mat = sc_utils::makeRingkeyFromScancontext(desc);
      std::vector<float> ring_key = sc_utils::eig2stdvec(ring_key_mat);

      entries.push_back({poses[i], desc, ring_key});

      if ((i + 1) % 50 == 0 || i == poses.size() - 1) {
        RCLCPP_INFO(get_logger(), "Processed %zu / %zu scans", i + 1, poses.size());
      }
    }

    RCLCPP_INFO(get_logger(), "Real scan entries: %zu (from %zu poses)", entries.size(), poses.size());
    return entries;
  }

  /**
   * 模式2: 从全局地图虚拟视点构建 SC 数据库 (fallback)
   */
  std::vector<Entry> build_from_virtual_viewpoints(const std::string& input, double radius) {
    RCLCPP_INFO(get_logger(), "=== Virtual Viewpoint Mode ===");

    double grid_step = get_parameter("grid_step").as_double();
    double min_density_radius = get_parameter("min_density_radius").as_double();
    int min_points = get_parameter("min_points_in_view").as_int();

    RCLCPP_INFO(get_logger(), "Loading %s ...", input.c_str());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(input, *cloud) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to load PCD: %s", input.c_str());
      return {};
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu points", cloud->size());

    float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (const auto& p : cloud->points) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }
    RCLCPP_INFO(get_logger(), "Map bounds: X[%.1f, %.1f] Y[%.1f, %.1f]", min_x, max_x, min_y, max_y);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<Eigen::Vector2f> viewpoints;
    for (double x = min_x; x <= max_x; x += grid_step) {
      for (double y = min_y; y <= max_y; y += grid_step) {
        pcl::PointXYZ query;
        query.x = x; query.y = y; query.z = 0;
        std::vector<int> indices;
        std::vector<float> dists;
        kdtree.radiusSearch(query, min_density_radius, indices, dists);
        if ((int)indices.size() >= min_points) {
          viewpoints.push_back(Eigen::Vector2f(x, y));
        }
      }
    }
    RCLCPP_INFO(get_logger(), "Generated %zu viewpoints (grid_step=%.1fm)", viewpoints.size(), grid_step);

    if (viewpoints.empty()) {
      RCLCPP_ERROR(get_logger(), "No valid viewpoints!");
      return {};
    }

    std::vector<Entry> entries;
    for (size_t i = 0; i < viewpoints.size(); i++) {
      const auto& vp = viewpoints[i];

      pcl::PointXYZ query;
      query.x = vp.x(); query.y = vp.y(); query.z = 0;
      std::vector<int> indices;
      std::vector<float> dists;
      kdtree.radiusSearch(query, radius, indices, dists);

      if ((int)indices.size() < min_points) continue;

      pcl::PointCloud<pcl::PointXYZ> local_cloud;
      local_cloud.reserve(indices.size());
      for (int idx : indices) {
        pcl::PointXYZ p;
        p.x = cloud->points[idx].x - vp.x();
        p.y = cloud->points[idx].y - vp.y();
        p.z = cloud->points[idx].z;
        local_cloud.push_back(p);
      }

      Eigen::MatrixXd desc = sc_utils::makeScancontext(local_cloud, radius);
      Eigen::MatrixXd ring_key_mat = sc_utils::makeRingkeyFromScancontext(desc);
      std::vector<float> ring_key = sc_utils::eig2stdvec(ring_key_mat);

      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      pose(0, 3) = vp.x();
      pose(1, 3) = vp.y();

      entries.push_back({pose, desc, ring_key});
    }

    RCLCPP_INFO(get_logger(), "Virtual viewpoint entries: %zu", entries.size());
    return entries;
  }

  void write_database(const std::string& output, const std::vector<Entry>& entries, double radius) {
    std::ofstream ofs(output, std::ios::binary);
    if (!ofs) {
      RCLCPP_ERROR(get_logger(), "Failed to open output: %s", output.c_str());
      return;
    }

    SCDatabaseHeader header;
    std::memcpy(header.magic, "SCDB_V01", 8);
    header.count = entries.size();
    header.num_ring = sc_utils::PC_NUM_RING;
    header.num_sector = sc_utils::PC_NUM_SECTOR;
    header.max_radius = radius;
    std::memset(header.reserved, 0, sizeof(header.reserved));
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto& entry : entries) {
      double pose_data[12];
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
          pose_data[r * 4 + c] = entry.pose(r, c);
      ofs.write(reinterpret_cast<const char*>(pose_data), sizeof(pose_data));

      std::vector<double> desc_data(sc_utils::PC_NUM_RING * sc_utils::PC_NUM_SECTOR);
      for (int r = 0; r < sc_utils::PC_NUM_RING; r++)
        for (int c = 0; c < sc_utils::PC_NUM_SECTOR; c++)
          desc_data[r * sc_utils::PC_NUM_SECTOR + c] = entry.desc(r, c);
      ofs.write(reinterpret_cast<const char*>(desc_data.data()), desc_data.size() * sizeof(double));

      ofs.write(reinterpret_cast<const char*>(entry.ring_key.data()), entry.ring_key.size() * sizeof(float));
    }

    ofs.close();
    std::ifstream size_check(output, std::ios::binary | std::ios::ate);
    double file_kb = size_check.tellg() / 1024.0;
    RCLCPP_INFO(get_logger(), "Database written to %s (%zu entries, %.1f KB)",
                output.c_str(), entries.size(), file_kb);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BuildSCDatabaseNode>();
  return 0;
}
