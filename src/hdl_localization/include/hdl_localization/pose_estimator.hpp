#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <boost/optional.hpp>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
namespace kkl {
  namespace alg {
template<typename T, class System> class UnscentedKalmanFilterX;
  }
}

namespace hdl_localization {

class PoseSystem;
class OdomSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration = 1.0);
  
  /**
   * @brief constructor with node pointer
   * @param node                rclcpp node pointer
   */
  PoseEstimator(rclcpp::Node* node);
  
  ~PoseEstimator();

  /**
   * @brief predict
   * @param stamp    timestamp
   */
  void predict(const rclcpp::Time& stamp);

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro);

  /**
   * @brief update the state of the odomety-based pose estimation
   */
  void predict_odom(const Eigen::Matrix4f& odom_delta);

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud);

  /* getters */
  rclcpp::Time last_correction_time() const;

  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;

  Eigen::Vector3f odom_pos() const;
  Eigen::Quaternionf odom_quat() const;
  Eigen::Matrix4f odom_matrix() const;

  const boost::optional<Eigen::Matrix4f>& wo_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& imu_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& odom_prediction_error() const;

  /**
   * @brief set initial pose from PoseWithCovarianceStamped message
   */
  void set_initial_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg);

  /**
   * @brief set odometry message
   */
  void set_odom(const nav_msgs::msg::Odometry::SharedPtr odom_msg);

  /**
   * @brief set global map for NDT target (registration->setInputTarget)
   */
  void set_global_map(const pcl::PointCloud<PointT>::ConstPtr& cloud);

  /**
   * @brief process point cloud
   */
  void process(const pcl::PointCloud<PointT>::ConstPtr& cloud, const rclcpp::Time& stamp);

  /**
   * @brief get pose message
   */
  geometry_msgs::msg::PoseWithCovarianceStamped get_pose_msg() const;

  /**
   * @brief get odometry message
   */
  nav_msgs::msg::Odometry get_odom_msg() const;

  /**
   * @brief check if aligned points are available
   */
  bool has_aligned_points() const;

  /**
   * @brief get aligned points
   */
  pcl::PointCloud<PointT>::Ptr get_aligned_points() const;

  bool registration_has_converged() const;
  Eigen::Matrix4f get_registration_final_transformation() const;
  void compute_inlier_stats(const pcl::PointCloud<PointT>::ConstPtr& aligned,
    double max_correspondence_dist, double max_valid_point_dist,
    float& matching_error, float& inlier_fraction) const;

private:
  rclcpp::Time init_stamp;             // when the estimator was initialized
  rclcpp::Time prev_stamp;             // when the estimator was updated last time
  rclcpp::Time last_correction_stamp;  // when the estimator performed the correction step
  double cool_time_duration;        //

  Eigen::MatrixXf process_noise;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>> odom_ukf;

  Eigen::Matrix4f last_observation;
  boost::optional<Eigen::Matrix4f> wo_pred_error;
  boost::optional<Eigen::Matrix4f> imu_pred_error;
  boost::optional<Eigen::Matrix4f> odom_pred_error;

  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::PointCloud<PointT>::Ptr last_aligned_;
  nav_msgs::msg::Odometry::SharedPtr prev_odom_;
  };

}  // namespace hdl_localization

#endif  // POSE_ESTIMATOR_HPP
