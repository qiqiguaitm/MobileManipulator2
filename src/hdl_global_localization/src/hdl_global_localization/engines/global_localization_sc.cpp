#include <hdl_global_localization/engines/global_localization_sc.hpp>

#include <fstream>
#include <cstring>

namespace hdl_global_localization {

// Must match build_sc_database.cpp
struct SCDatabaseHeader {
  char magic[8];
  uint32_t count;
  uint32_t num_ring;
  uint32_t num_sector;
  double max_radius;
  char reserved[32];
};

GlobalLocalizationSC::GlobalLocalizationSC(rclcpp::Node::SharedPtr node) : node_(node) {
  sc_database_path_ = node_->declare_parameter<std::string>("sc_database_path", "");
  sc_dist_threshold_ = node_->declare_parameter<double>("sc_dist_threshold", 0.4);
  sc_num_candidates_ = node_->declare_parameter<int>("sc_num_candidates", 3);
  sc_max_radius_ = node_->declare_parameter<double>("sc_max_radius", 20.0);

  RCLCPP_INFO(node_->get_logger(), "ScanContext engine initialized (threshold=%.2f, candidates=%d, radius=%.1f)",
              sc_dist_threshold_, sc_num_candidates_, sc_max_radius_);
}

GlobalLocalizationSC::~GlobalLocalizationSC() {}

void GlobalLocalizationSC::set_global_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  // Load pre-built SC database
  if (sc_database_path_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "sc_database_path not set!");
    return;
  }

  if (!load_database(sc_database_path_)) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to load SC database: %s", sc_database_path_.c_str());
    return;
  }

  // Build KD-tree from ring keys
  ring_keys_mat_.clear();
  ring_keys_mat_.reserve(entries_.size());
  for (const auto& entry : entries_) {
    ring_keys_mat_.push_back(entry.ring_key);
  }

  ring_key_tree_.reset();
  ring_key_tree_ = std::make_unique<InvKeyTree>(
    sc_utils::PC_NUM_RING, ring_keys_mat_, 10);

  RCLCPP_INFO(node_->get_logger(), "SC database loaded: %zu entries, KD-tree built", entries_.size());
}

GlobalLocalizationResults GlobalLocalizationSC::query(
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, int max_num_candidates) {

  std::vector<GlobalLocalizationResult::Ptr> results;

  if (!ring_key_tree_ || entries_.empty()) {
    RCLCPP_WARN(node_->get_logger(), "SC database not loaded");
    return GlobalLocalizationResults(results);
  }

  if (!cloud || cloud->empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty query cloud");
    return GlobalLocalizationResults(results);
  }

  // Compute SC descriptor for query cloud
  Eigen::MatrixXd query_desc = sc_utils::makeScancontext(*cloud, sc_max_radius_);
  Eigen::MatrixXd query_ring_key = sc_utils::makeRingkeyFromScancontext(query_desc);
  std::vector<float> query_ring_vec = sc_utils::eig2stdvec(query_ring_key);

  // KD-tree search for top-K candidates by ring key
  int k = std::min(sc_num_candidates_, (int)entries_.size());
  std::vector<size_t> candidate_indices(k);
  std::vector<float> candidate_dists(k);

  nanoflann::KNNResultSet<float> knn_result(k);
  knn_result.init(&candidate_indices[0], &candidate_dists[0]);
  ring_key_tree_->index->findNeighbors(knn_result, &query_ring_vec[0], nanoflann::SearchParams(10));

  // Pairwise SC distance for each candidate, collect all passing threshold
  struct SCCandidate {
    int entry_idx;
    double dist;
    int shift;
  };
  std::vector<SCCandidate> candidates;

  for (int i = 0; i < k; i++) {
    size_t idx = candidate_indices[i];
    auto [dist, shift] = sc_utils::distanceBtnScanContext(query_desc, entries_[idx].descriptor);

    RCLCPP_INFO(node_->get_logger(), "SC candidate %zu: dist=%.3f, shift=%d, pos=(%.2f, %.2f)",
                idx, dist, shift, entries_[idx].pose(0, 3), entries_[idx].pose(1, 3));

    if (dist <= sc_dist_threshold_) {
      candidates.push_back({(int)idx, dist, shift});
    }
  }

  if (candidates.empty()) {
    RCLCPP_WARN(node_->get_logger(), "SC matching failed: no candidate below threshold=%.3f", sc_dist_threshold_);
    return GlobalLocalizationResults(results);
  }

  // Sort by distance (best first)
  std::sort(candidates.begin(), candidates.end(),
            [](const SCCandidate& a, const SCCandidate& b) { return a.dist < b.dist; });

  // Return up to max_num_candidates results
  int num_results = std::min((int)candidates.size(), max_num_candidates);
  for (int i = 0; i < num_results; i++) {
    const auto& cand = candidates[i];
    const auto& entry = entries_[cand.entry_idx];

    // Compute yaw from column shift
    double yaw_offset = static_cast<double>(cand.shift) * (360.0 / sc_utils::PC_NUM_SECTOR) * M_PI / 180.0;

    // Build 3D pose from viewpoint + yaw
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translation() = Eigen::Vector3f(entry.pose(0, 3), entry.pose(1, 3), 0.0f);
    Eigen::AngleAxisf yaw_rot(yaw_offset, Eigen::Vector3f::UnitZ());
    pose.linear() = yaw_rot.toRotationMatrix();

    double inlier_fraction = 1.0 - cand.dist;
    results.push_back(std::make_shared<GlobalLocalizationResult>(cand.dist, inlier_fraction, pose));

    RCLCPP_INFO(node_->get_logger(), "SC result[%d]: entry=%d, dist=%.3f, yaw=%.1fdeg, pos=(%.2f, %.2f)",
                i, cand.entry_idx, cand.dist, yaw_offset * 180.0 / M_PI,
                pose.translation().x(), pose.translation().y());
  }

  return GlobalLocalizationResults(results);
}

bool GlobalLocalizationSC::load_database(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    RCLCPP_ERROR(node_->get_logger(), "Cannot open SC database: %s", path.c_str());
    return false;
  }

  SCDatabaseHeader header;
  ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

  if (std::strncmp(header.magic, "SCDB_V01", 8) != 0) {
    RCLCPP_ERROR(node_->get_logger(), "Invalid SC database magic");
    return false;
  }

  RCLCPP_INFO(node_->get_logger(), "SC database: %u entries, %ux%u, radius=%.1f",
              header.count, header.num_ring, header.num_sector, header.max_radius);

  entries_.clear();
  entries_.reserve(header.count);

  for (uint32_t i = 0; i < header.count; i++) {
    SCEntry entry;

    // Read pose (3x4 row-major)
    double pose_data[12];
    ifs.read(reinterpret_cast<char*>(pose_data), sizeof(pose_data));
    entry.pose = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 4; c++)
        entry.pose(r, c) = pose_data[r * 4 + c];

    // Read descriptor (row-major)
    std::vector<double> desc_data(header.num_ring * header.num_sector);
    ifs.read(reinterpret_cast<char*>(desc_data.data()), desc_data.size() * sizeof(double));
    entry.descriptor = Eigen::MatrixXd(header.num_ring, header.num_sector);
    for (uint32_t r = 0; r < header.num_ring; r++)
      for (uint32_t c = 0; c < header.num_sector; c++)
        entry.descriptor(r, c) = desc_data[r * header.num_sector + c];

    // Read ring key
    entry.ring_key.resize(header.num_ring);
    ifs.read(reinterpret_cast<char*>(entry.ring_key.data()), header.num_ring * sizeof(float));

    entries_.push_back(std::move(entry));
  }

  return true;
}

}  // namespace hdl_global_localization
