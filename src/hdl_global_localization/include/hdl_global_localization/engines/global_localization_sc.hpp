#ifndef HDL_GLOBAL_LOCALIZATION_SC_HPP
#define HDL_GLOBAL_LOCALIZATION_SC_HPP

#include <vector>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <hdl_global_localization/engines/global_localization_engine.hpp>
#include <hdl_global_localization/sc/sc_utils.hpp>
#include <hdl_global_localization/sc/nanoflann.hpp>
#include <hdl_global_localization/sc/KDTreeVectorOfVectorsAdaptor.h>

namespace hdl_global_localization {

using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

class GlobalLocalizationSC : public GlobalLocalizationEngine {
public:
  GlobalLocalizationSC(rclcpp::Node::SharedPtr node);
  ~GlobalLocalizationSC() override;

  void set_global_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) override;
  GlobalLocalizationResults query(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, int max_num_candidates) override;

private:
  bool load_database(const std::string& path);

  rclcpp::Node::SharedPtr node_;

  // Parameters
  std::string sc_database_path_;
  double sc_dist_threshold_;
  int sc_num_candidates_;
  double sc_max_radius_;

  // Database entries
  struct SCEntry {
    Eigen::Matrix4d pose;         // 4x4 viewpoint pose
    Eigen::MatrixXd descriptor;   // 20x60 SC descriptor
    std::vector<float> ring_key;  // 20-dim ring key
  };
  std::vector<SCEntry> entries_;

  // KD-tree for ring key search
  KeyMat ring_keys_mat_;
  std::unique_ptr<InvKeyTree> ring_key_tree_;
};

}  // namespace hdl_global_localization

#endif  // HDL_GLOBAL_LOCALIZATION_SC_HPP
