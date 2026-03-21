// ROS2版本 - HDL Localization Standalone Node
// 完整从 ROS1 hdl_localization_nodelet.cpp 迁移

#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pclomp/ndt_omp.h>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/msg/scan_matching_status.hpp>
#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

namespace hdl_localization {

class HdlLocalizationNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : rclcpp::Node("hdl_localization", options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_),
      tf_broadcaster_(this)
  {
    initialize_params();

    robot_odom_frame_id_ = this->declare_parameter<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id_ = this->declare_parameter<std::string>("odom_child_frame_id", "base_link");

    use_imu_ = this->declare_parameter<bool>("use_imu", true);
    invert_acc_ = this->declare_parameter<bool>("invert_acc", false);
    invert_gyro_ = this->declare_parameter<bool>("invert_gyro", false);

    if (use_imu_) {
      RCLCPP_INFO(this->get_logger(), "enable imu-based prediction");
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
          "imu", 256,
          std::bind(&HdlLocalizationNode::imu_callback, this, std::placeholders::_1));
    }

    points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", 5,
        std::bind(&HdlLocalizationNode::points_callback, this, std::placeholders::_1));
    // globalmap 使用 transient_local QoS 以匹配 globalmap_publisher（latched）
    rclcpp::QoS globalmap_qos(1);
    globalmap_qos.transient_local();
    globalmap_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "globalmap", globalmap_qos,
        std::bind(&HdlLocalizationNode::globalmap_callback, this, std::placeholders::_1));
    initialpose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "initialpose", 8,
        std::bind(&HdlLocalizationNode::initialpose_callback, this, std::placeholders::_1));

    pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("pose", 5);
    aligned_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_points", 5);
    status_pub_ = this->create_publisher<hdl_localization::msg::ScanMatchingStatus>("status", 5);
    localization_status_pub_ = this->create_publisher<std_msgs::msg::Bool>("localization_status", 5);

    // global localization
    use_global_localization_ = this->declare_parameter<bool>("use_global_localization", true);
    if(use_global_localization_) {
      RCLCPP_INFO(this->get_logger(), "wait for global localization services");
      // 尝试多个可能的服务名（与 globalmap_publisher.py 一致）
      set_global_map_client_ = this->create_client<hdl_global_localization::srv::SetGlobalMap>(
          "/set_global_map");
      query_global_localization_client_ = this->create_client<hdl_global_localization::srv::QueryGlobalLocalization>(
          "/query");
      relocalize_server_ = this->create_service<std_srvs::srv::Empty>(
          "/relocalize",
          std::bind(&HdlLocalizationNode::relocalize, this, std::placeholders::_1, std::placeholders::_2));

      // 启动定时器异步等待服务可用（避免阻塞构造函数和回调）
      // 使用100ms间隔，确保在点云到达前服务已准备好
      global_service_ready_ = false;
      service_check_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(1000),
          [this]() {
            if(global_service_ready_) {
              return;  // 已准备好，不再检查
            }
            // 使用 wait_for_service 替代 service_is_ready，更可靠
            bool set_map_ready = set_global_map_client_->wait_for_service(std::chrono::milliseconds(10));
            bool query_ready = query_global_localization_client_->wait_for_service(std::chrono::milliseconds(10));
            RCLCPP_INFO(this->get_logger(), "Checking services: set_global_map=%d, query=%d", set_map_ready, query_ready);
            if(set_map_ready && query_ready) {
              RCLCPP_INFO(this->get_logger(), "Global localization services are ready!");
              global_service_ready_ = true;

              // 发送挂起的请求（如果有）
              if(pending_globalmap_req_) {
                RCLCPP_INFO(this->get_logger(), "Sending pending set_global_map request...");
                auto result = set_global_map_client_->async_send_request(pending_globalmap_req_);
                pending_globalmap_req_.reset();
                RCLCPP_INFO(this->get_logger(), "set_global_map request sent");
              }

              service_check_timer_->cancel();  // 停止定时器
            }
          });
    }
  }

  virtual ~HdlLocalizationNode() {}

private:
  void initialize_params() {
    // 降采样方法
    auto downsample_method = this->declare_parameter<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);

    if(downsample_method == "VOXELGRID") {
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter_.reset(voxelgrid);
    } else {
      RCLCPP_ERROR(this->get_logger(), "unknown downsample_method: %s", downsample_method.c_str());
      downsample_filter_.reset(new pcl::VoxelGrid<PointT>());
    }

    RCLCPP_INFO(this->get_logger(), "create registration method");
    registration_ = create_registration();

    // 初始化delta估计器
    delta_estimater_.reset(new DeltaEstimater(registration_));

    // 声明其他参数
    this->declare_parameter<double>("cool_time_duration", 2.0);
    this->declare_parameter<bool>("enable_robot_odometry_prediction", false);
    this->declare_parameter<double>("status_max_correspondence_dist", 0.2);
    this->declare_parameter<double>("localization_inlier_threshold", 0.90);

    // 参数化初始位姿
    this->declare_parameter<bool>("specify_init_pose", false);
    this->declare_parameter<double>("init_pos_x", 0.0);
    this->declare_parameter<double>("init_pos_y", 0.0);
    this->declare_parameter<double>("init_pos_z", 0.0);
    this->declare_parameter<double>("init_ori_w", 1.0);
    this->declare_parameter<double>("init_ori_x", 0.0);
    this->declare_parameter<double>("init_ori_y", 0.0);
    this->declare_parameter<double>("init_ori_z", 0.0);

    // 自动重定位参数
    this->declare_parameter<bool>("auto_relocalization", false);
    this->declare_parameter<int>("auto_reloc_delay_ms", 3000);
    this->declare_parameter<double>("auto_reloc_conf_threshold", 0.4);
    this->declare_parameter<int>("auto_reloc_ndt_candidates", 6);

    relocalizing_ = false;
  }

  pcl::Registration<PointT, PointT>::Ptr create_registration() {
    std::string reg_method = this->declare_parameter<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = this->declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = this->declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = this->declare_parameter<double>("ndt_resolution", 1.0);
    int ndt_max_iterations = this->declare_parameter<int>("ndt_max_iterations", 35);

    if(reg_method == "NDT_OMP") {
      RCLCPP_INFO(this->get_logger(), "NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      ndt->setMaximumIterations(ndt_max_iterations);
      RCLCPP_INFO(this->get_logger(), "NDT resolution: %.2f, max_iterations: %d", ndt_resolution, ndt_max_iterations);

      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(this->get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(this->get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          RCLCPP_INFO(this->get_logger(), "search_method KDTREE is selected");
        } else {
          RCLCPP_WARN(this->get_logger(), "invalid search method was given");
          RCLCPP_WARN(this->get_logger(), "default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    }

    RCLCPP_ERROR(this->get_logger(), "unknown registration method: %s", reg_method.c_str());
    return nullptr;
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter_) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter_->setInputCloud(cloud);
    downsample_filter_->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex_);
    imu_data_.push_back(imu_msg);
  }

  void points_callback(const sensor_msgs::msg::PointCloud2::SharedPtr points_msg) {
    if(!globalmap_) {
      RCLCPP_ERROR(this->get_logger(), "globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if(pcl_cloud->empty()) {
      RCLCPP_ERROR(this->get_logger(), "cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    try {
      if(tf_buffer_.canTransform(odom_child_frame_id_, pcl_cloud->header.frame_id, stamp, rclcpp::Duration::from_seconds(0.1))) {
        sensor_msgs::msg::PointCloud2 transformed_msg;
        auto transform = tf_buffer_.lookupTransform(odom_child_frame_id_, pcl_cloud->header.frame_id, stamp, rclcpp::Duration::from_seconds(0.1));
        tf2::doTransform(*points_msg, transformed_msg, transform);
        pcl::fromROSMsg(transformed_msg, *cloud);
      } else {
        RCLCPP_ERROR(this->get_logger(), "cannot transform point cloud to %s", odom_child_frame_id_.c_str());
        return;
      }
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
      return;
    }

    auto filtered = downsample(cloud);
    last_scan_ = filtered;

    if(relocalizing_) {
      delta_estimater_->add_frame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex_);
    if(!pose_estimator_) {
      // 与ROS1一致：等待用户手动设置initialpose或调用/relocalize服务
      // 不自动触发全局定位，因为BBS粗定位精度有限
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "waiting for initial pose input!! Please set initialpose in RViz or call /relocalize service. "
          "(globalmap=%d, service_ready=%d, relocalizing=%d)",
          globalmap_ != nullptr, global_service_ready_.load(), relocalizing_.load());
      return;
    }

    // predict
    if(!use_imu_) {
      pose_estimator_->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex_);
      auto imu_iter = imu_data_.begin();
      for(; imu_iter != imu_data_.end(); imu_iter++) {
        if(rclcpp::Time(stamp) < rclcpp::Time((*imu_iter)->header.stamp)) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc_ ? -1.0 : 1.0;
        double gyro_sign = invert_gyro_ ? -1.0 : 1.0;
        pose_estimator_->predict((*imu_iter)->header.stamp,
                                 acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z),
                                 gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data_.erase(imu_data_.begin(), imu_iter);
    }

    // odometry-based prediction
    rclcpp::Time last_correction_time = pose_estimator_->last_correction_time();
    bool enable_odom_prediction = this->get_parameter("enable_robot_odometry_prediction").as_bool();

    if(enable_odom_prediction && last_correction_time.nanoseconds() != 0) {
      geometry_msgs::msg::TransformStamped odom_delta;
      bool got_transform = false;

      try {
        if(tf_buffer_.canTransform(odom_child_frame_id_, last_correction_time, odom_child_frame_id_, stamp,
                                    robot_odom_frame_id_, rclcpp::Duration::from_seconds(0.1))) {
          odom_delta = tf_buffer_.lookupTransform(odom_child_frame_id_, last_correction_time,
                                                   odom_child_frame_id_, stamp, robot_odom_frame_id_);
          got_transform = true;
        } else if(tf_buffer_.canTransform(odom_child_frame_id_, last_correction_time, odom_child_frame_id_,
                                           rclcpp::Time(0), robot_odom_frame_id_, rclcpp::Duration::from_seconds(0.0))) {
          odom_delta = tf_buffer_.lookupTransform(odom_child_frame_id_, last_correction_time,
                                                   odom_child_frame_id_, rclcpp::Time(0), robot_odom_frame_id_);
          got_transform = true;
        }
      } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_STREAM(this->get_logger(), "failed to look up transform: " << ex.what());
      }

      if(got_transform && odom_delta.header.stamp.sec != 0) {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator_->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    auto aligned = pose_estimator_->correct(stamp, filtered);

    // --- localization_status: subsampled inlier check ---
    {
      double max_corr_dist = 0.5;
      this->get_parameter_or("status_max_correspondence_dist", max_corr_dist, 0.5);
      double threshold = 0.3;
      this->get_parameter_or("localization_inlier_threshold", threshold, 0.3);
      double max_dist_sq = max_corr_dist * max_corr_dist;

      int step = std::max(1, (int)aligned->size() / 500);
      int inlier = 0, checked = 0;
      std::vector<int> nn_idx(1);
      std::vector<float> nn_dist(1);
      for (size_t i = 0; i < aligned->size(); i += step) {
        const auto& pt = aligned->at(i);
        if (registration_->getSearchMethodTarget()->nearestKSearch(pt, 1, nn_idx, nn_dist) > 0) {
          checked++;
          if (nn_dist[0] < max_dist_sq) {
            inlier++;
          }
        }
      }
      bool ok = registration_->hasConverged()
             && checked > 0
             && (double)inlier / checked > threshold;

      std_msgs::msg::Bool status_msg;
      status_msg.data = ok;
      localization_status_pub_->publish(status_msg);
    }

    if(aligned_pub_->get_subscription_count()) {
      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(*aligned, aligned_msg);
      aligned_msg.header.frame_id = "map";
      aligned_msg.header.stamp = points_msg->header.stamp;
      aligned_pub_->publish(aligned_msg);
    }

    if(status_pub_->get_subscription_count()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    publish_odometry(points_msg->header.stamp, pose_estimator_->matrix());
  }

  void globalmap_callback(const sensor_msgs::msg::PointCloud2::SharedPtr points_msg) {
    RCLCPP_INFO(this->get_logger(), "globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap_ = cloud;

    registration_->setInputTarget(globalmap_);

    if(use_global_localization_) {
      RCLCPP_INFO(this->get_logger(), "set globalmap for global localization!");
      auto req = std::make_shared<hdl_global_localization::srv::SetGlobalMap::Request>();
      pcl::toROSMsg(*globalmap_, req->global_map);

      // 检查服务是否已通过后台定时器发现
      if(!global_service_ready_) {
        // 立即检查一次（可能刚好准备好了）
        if(set_global_map_client_->service_is_ready() &&
           query_global_localization_client_->service_is_ready()) {
          RCLCPP_INFO(this->get_logger(), "Global localization services just became ready!");
          global_service_ready_ = true;
          if(service_check_timer_) {
            service_check_timer_->cancel();
          }
        } else {
          // 服务还没准备好，保存请求等定时器发送
          RCLCPP_INFO(this->get_logger(), "Global localization service not ready yet, will retry...");
          pending_globalmap_req_ = req;
          return;
        }
      }

      auto result = set_global_map_client_->async_send_request(req);
      RCLCPP_INFO(this->get_logger(), "set_global_map request sent");
    }

    // 尝试从参数自动初始化位姿 (specify_init_pose)
    if (try_auto_init_from_params()) {
      // specify_init_pose 成功，不需要自动重定位
    } else if (this->get_parameter("auto_relocalization").as_bool() && !auto_reloc_triggered_) {
      // 延迟触发自动重定位 (等待足够的 scan 数据)
      int delay_ms = this->get_parameter("auto_reloc_delay_ms").as_int();
      RCLCPP_INFO(this->get_logger(), "Auto-relocalization enabled, will trigger in %dms", delay_ms);
      auto_reloc_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(delay_ms),
        [this]() {
          auto_reloc_timer_->cancel();
          trigger_auto_relocalization();
        });
    }
  }

  void relocalize(
      const std::shared_ptr<std_srvs::srv::Empty::Request> req,
      std::shared_ptr<std_srvs::srv::Empty::Response> res)
  {
    RCLCPP_INFO(this->get_logger(), "relocalize");

    if(!last_scan_) {
      RCLCPP_WARN(this->get_logger(), "no scan has been received");
      return;
    }

    relocalizing_ = true;
    delta_estimater_->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan_;

    auto req_global = std::make_shared<hdl_global_localization::srv::QueryGlobalLocalization::Request>();
    pcl::toROSMsg(*scan, req_global->cloud);
    req_global->max_num_candidates = 1;

    // 检查服务是否可用
    if(!global_service_ready_) {
      RCLCPP_WARN(this->get_logger(), "Global localization services not ready yet");
      relocalizing_ = false;
      return;
    }

    // 异步发送请求并设置回调 - 避免在已spinning节点中再次spin
    auto future = query_global_localization_client_->async_send_request(
        req_global,
        [this](rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedFuture future) {
          this->handle_global_localization_result(future);
        });
  }

  void handle_global_localization_result(
      rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedFuture future)
  {
    try {
      auto result = future.get();

      if(result->poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "global localization failed");
        relocalizing_ = false;
        return;
      }

      const auto& result_pose = result->poses[0];
      RCLCPP_INFO(this->get_logger(), "--- Global localization result ---");
      RCLCPP_INFO(this->get_logger(), "Trans : %.3f %.3f %.3f",
                  result_pose.position.x, result_pose.position.y, result_pose.position.z);
      RCLCPP_INFO(this->get_logger(), "Quat  : %.3f %.3f %.3f %.3f",
                  result_pose.orientation.w, result_pose.orientation.x,
                  result_pose.orientation.y, result_pose.orientation.z);
      RCLCPP_INFO(this->get_logger(), "Error : %.3f", result->errors[0]);
      RCLCPP_INFO(this->get_logger(), "Inlier: %.3f", result->inlier_fractions[0]);

      Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
      pose.linear() = Eigen::Quaternionf(result_pose.orientation.w, result_pose.orientation.x,
                                         result_pose.orientation.y, result_pose.orientation.z).toRotationMatrix();
      pose.translation() = Eigen::Vector3f(result_pose.position.x, result_pose.position.y, result_pose.position.z);

      // 应用全局定位期间的运动增量（与ROS1版本一致）
      pose = pose * delta_estimater_->estimated_delta();

      // 设置初始位姿
      std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
      pose_estimator_.reset(new hdl_localization::PoseEstimator(
        registration_,
        pose.translation(),
        Eigen::Quaternionf(pose.linear()),
        this->get_parameter("cool_time_duration").as_double()));
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "global localization request failed: %s", e.what());
    }

    relocalizing_ = false;
  }

  /**
   * @brief 自动重定位: 在 globalmap 到达后延迟触发全局定位查询
   */
  void trigger_auto_relocalization() {
    if (auto_reloc_triggered_) return;
    auto_reloc_triggered_ = true;

    // 检查前置条件
    if (!globalmap_) {
      RCLCPP_WARN(this->get_logger(), "Auto-reloc: no globalmap yet");
      auto_reloc_triggered_ = false;
      return;
    }
    if (!global_service_ready_) {
      RCLCPP_WARN(this->get_logger(), "Auto-reloc: global localization service not ready");
      auto_reloc_triggered_ = false;
      return;
    }
    if (!last_scan_) {
      RCLCPP_WARN(this->get_logger(), "Auto-reloc: no scan available yet, will retry in 2s");
      auto_reloc_triggered_ = false;
      auto_reloc_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(2000),
        [this]() {
          auto_reloc_timer_->cancel();
          trigger_auto_relocalization();
        });
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Auto-reloc: triggering global localization query (multi-candidate + NDT validation)...");
    relocalizing_ = true;
    delta_estimater_->reset();

    auto req = std::make_shared<hdl_global_localization::srv::QueryGlobalLocalization::Request>();
    pcl::toROSMsg(*last_scan_, req->cloud);
    req->max_num_candidates = this->get_parameter("auto_reloc_ndt_candidates").as_int();  // 参数化: 多候选 NDT 验证

    auto future = query_global_localization_client_->async_send_request(
      req,
      [this](rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedFuture future) {
        try {
          auto result = future.get();
          if (result->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Auto-reloc: no result, will retry once in 5s");
            relocalizing_ = false;
            auto_reloc_triggered_ = false;
            auto_reloc_timer_ = this->create_wall_timer(
              std::chrono::milliseconds(5000),
              [this]() {
                auto_reloc_timer_->cancel();
                trigger_auto_relocalization();
              });
            return;
          }

          // NDT 验证: 多信号组合评分选最优候选
          // SC距离=地点识别信号, inlier=几何验证, NDT漂移=局部极值检测
          double best_score = -1.0;
          double best_inlier = -1.0;
          int best_candidate = -1;
          Eigen::Isometry3f best_pose = Eigen::Isometry3f::Identity();

          RCLCPP_INFO(this->get_logger(), "Auto-reloc: %zu candidates, NDT-validating each...",
                      result->poses.size());

          // 构建 KD-tree 一次，所有候选共用
          pcl::KdTreeFLANN<PointT> kdtree;
          kdtree.setInputCloud(globalmap_);
          double max_corr_dist = this->get_parameter("status_max_correspondence_dist").as_double();
          double max_corr_dist_sq = max_corr_dist * max_corr_dist;

          for (size_t i = 0; i < result->poses.size(); i++) {
            const auto& p = result->poses[i];
            double sc_dist = result->errors[i];

            Eigen::Isometry3f candidate_pose = Eigen::Isometry3f::Identity();
            candidate_pose.linear() = Eigen::Quaternionf(
              p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).toRotationMatrix();
            candidate_pose.translation() = Eigen::Vector3f(p.position.x, p.position.y, p.position.z);

            // NDT 验证: 用候选位姿作为初始猜测，跑 NDT 对齐
            pcl::PointCloud<PointT> aligned;
            registration_->setInputSource(last_scan_);
            registration_->align(aligned, candidate_pose.matrix().cast<float>());

            if (!registration_->hasConverged()) {
              RCLCPP_DEBUG(this->get_logger(), "  Candidate[%zu]: NDT not converged, skip (sc_dist=%.3f)", i, sc_dist);
              continue;
            }

            // 计算 inlier_fraction
            int inlier_count = 0;
            pcl::PointCloud<PointT>::Ptr aligned_ptr(new pcl::PointCloud<PointT>(aligned));
            for (const auto& pt : aligned_ptr->points) {
              std::vector<int> nn_idx(1);
              std::vector<float> nn_dist(1);
              if (kdtree.nearestKSearch(pt, 1, nn_idx, nn_dist) > 0 && nn_dist[0] < max_corr_dist_sq) {
                inlier_count++;
              }
            }
            double inlier_fraction = aligned_ptr->empty() ? 0.0 : (double)inlier_count / aligned_ptr->size();

            Eigen::Matrix4f ndt_result = registration_->getFinalTransformation();
            float ndt_score = registration_->getFitnessScore();

            // 计算 NDT 漂移: NDT 精配准把位姿移动了多远 (2D)
            Eigen::Vector2f sc_pos_2d(candidate_pose.translation().x(), candidate_pose.translation().y());
            Eigen::Vector2f ndt_pos_2d(ndt_result(0, 3), ndt_result(1, 3));
            double drift = (ndt_pos_2d - sc_pos_2d).norm();

            // 组合评分 (4信号融合):
            //   sc_confidence  : SC距离越小=地点识别越可信
            //   inlier_fraction: 几何对齐质量 (点云匹配比例)
            //   ndt_quality    : NDT适配度，1/(1+fitness)，惩罚均方距离大的候选
            //   drift_factor   : NDT漂移>1m时指数衰减，防止跑到错误局部极值
            double sc_confidence = 1.0 - sc_dist;
            double ndt_quality = 1.0 / (1.0 + ndt_score);   // [0,1]，fitness越小越好
            double drift_factor = std::exp(-std::max(0.0, drift - 1.0) / 2.0);
            double combined_score = sc_confidence * inlier_fraction * ndt_quality * drift_factor;

            RCLCPP_DEBUG(this->get_logger(),
              "  Candidate[%zu]: sc_dist=%.3f, ndt_fit=%.3f, ndt_q=%.2f, inlier=%.2f, drift=%.2f, score=%.4f, pos=(%.2f, %.2f)",
              i, sc_dist, ndt_score, ndt_quality, inlier_fraction, drift, combined_score,
              ndt_result(0, 3), ndt_result(1, 3));

            if (combined_score > best_score) {
              best_score = combined_score;
              best_inlier = inlier_fraction;
              best_candidate = i;
              best_pose = Eigen::Isometry3f::Identity();
              best_pose.linear() = ndt_result.block<3, 3>(0, 0);
              best_pose.translation() = ndt_result.block<3, 1>(0, 3);
            }
          }

          // 双重验证阈值: inlier + combined_score
          const double MIN_INLIER_FRACTION = 0.5;
          double min_score = this->get_parameter("auto_reloc_conf_threshold").as_double();

          if (best_candidate < 0 || best_inlier < MIN_INLIER_FRACTION || best_score < min_score) {
            std::string reason;
            if (best_candidate < 0) reason = "no candidate";
            else if (best_inlier < MIN_INLIER_FRACTION) reason = "inlier=" + std::to_string(best_inlier).substr(0,4) + "<" + std::to_string(MIN_INLIER_FRACTION).substr(0,4);
            else reason = "score=" + std::to_string(best_score).substr(0,6) + "<" + std::to_string(min_score).substr(0,6);

            RCLCPP_WARN(this->get_logger(),
              "Auto-reloc: failed (%s), please set initialpose in RViz", reason.c_str());
            relocalizing_ = false;
            return;
          }

          // 应用运动增量
          best_pose = best_pose * Eigen::Isometry3f(delta_estimater_->estimated_delta());

          std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
          pose_estimator_.reset(new hdl_localization::PoseEstimator(
            registration_,
            best_pose.translation(),
            Eigen::Quaternionf(best_pose.linear()),
            this->get_parameter("cool_time_duration").as_double()));

          RCLCPP_INFO(this->get_logger(),
            "Auto-reloc: SUCCESS! candidate[%d], score=%.4f, inlier=%.2f, pos=(%.2f, %.2f, %.2f)",
            best_candidate, best_score, best_inlier,
            best_pose.translation().x(), best_pose.translation().y(), best_pose.translation().z());

        } catch (const std::exception& e) {
          RCLCPP_WARN(this->get_logger(), "Auto-reloc failed: %s", e.what());
        }
        relocalizing_ = false;
      });
  }

  /**
   * @brief 从参数自动初始化位姿 (修复 specify_init_pose 死代码)
   * 在 globalmap 收到后调用，如果 specify_init_pose=true 且 pose_estimator_ 未创建
   * @return true if auto-init succeeded
   */
  bool try_auto_init_from_params() {
    if (!this->get_parameter("specify_init_pose").as_bool()) {
      return false;
    }

    std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
    if (pose_estimator_) {
      return false;  // 已经初始化过
    }

    double px = this->get_parameter("init_pos_x").as_double();
    double py = this->get_parameter("init_pos_y").as_double();
    double pz = this->get_parameter("init_pos_z").as_double();
    double ow = this->get_parameter("init_ori_w").as_double();
    double ox = this->get_parameter("init_ori_x").as_double();
    double oy = this->get_parameter("init_ori_y").as_double();
    double oz = this->get_parameter("init_ori_z").as_double();

    RCLCPP_INFO(this->get_logger(), "Auto-init from params: pos=(%.2f, %.2f, %.2f) quat=(%.3f, %.3f, %.3f, %.3f)",
                px, py, pz, ow, ox, oy, oz);

    pose_estimator_.reset(new hdl_localization::PoseEstimator(
      registration_,
      Eigen::Vector3f(px, py, pz),
      Eigen::Quaternionf(ow, ox, oy, oz),
      this->get_parameter("cool_time_duration").as_double()));

    return true;
  }

  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg) {
    RCLCPP_INFO(this->get_logger(), "initialpose received");

    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;

    std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
    pose_estimator_.reset(new hdl_localization::PoseEstimator(
      registration_,
      Eigen::Vector3f(p.x, p.y, p.z),
      Eigen::Quaternionf(q.w, q.x, q.y, q.z),
      this->get_parameter("cool_time_duration").as_double()));
  }

  void publish_odometry(const builtin_interfaces::msg::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    std::string error_msg;
    if(tf_buffer_.canTransform(robot_odom_frame_id_, odom_child_frame_id_, tf2::TimePointZero, &error_msg)) {
      try {
        geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = odom_child_frame_id_;
        map_wrt_frame.child_frame_id = "map";

        geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer_.lookupTransform(
          robot_odom_frame_id_, odom_child_frame_id_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.1));

        geometry_msgs::msg::TransformStamped map_wrt_odom;
        tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

        tf2::Transform odom_wrt_map;
        tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
        odom_wrt_map = odom_wrt_map.inverse();

        geometry_msgs::msg::TransformStamped odom_trans;
        odom_trans.transform = tf2::toMsg(odom_wrt_map);
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = robot_odom_frame_id_;

        tf_broadcaster_.sendTransform(odom_trans);
      } catch (tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "%s", ex.what());
      }
    } else {
      // 回退: 发布 map->odom (而非 map->base_link!)
      // 发布 map->base_link 会与 body->base_link 静态TF冲突，导致base_link有两个父节点，TF树断裂
      // 此处近似 map->odom ≈ map->base_link (当 odom->base_link ≈ identity，即Fast-LIO未启动时)
      geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id_;
      tf_broadcaster_.sendTransform(odom_trans);
    }

    // publish the odometry
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
    odom.child_frame_id = odom_child_frame_id_;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub_->publish(odom);
  }

  void publish_scan_matching_status(const std_msgs::msg::Header& header,
                                     pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    hdl_localization::msg::ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration_->hasConverged();
    status.matching_error = 0.0;

    // 使用 get_parameter_or 避免重复声明
    double max_correspondence_dist = 0.5;
    this->get_parameter_or("status_max_correspondence_dist", max_correspondence_dist, 0.5);

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration_->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        status.matching_error += k_sq_dists[0];
        num_inliers++;
      }
    }
    status.matching_error /= num_inliers;
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    // relative_pose留空或设置默认值
    status.relative_pose.rotation.w = 1.0;

    // prediction_labels和prediction_errors根据实际msg定义填充
    // 如果类型不匹配，暂时留空

    status_pub_->publish(status);
  }


private:
  // TF相关
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // 订阅/发布
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
  rclcpp::Publisher<hdl_localization::msg::ScanMatchingStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr localization_status_pub_;

  // 服务
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr relocalize_server_;
  rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedPtr set_global_map_client_;
  rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_client_;
  rclcpp::TimerBase::SharedPtr service_check_timer_;
  std::atomic_bool global_service_ready_{false};
  std::shared_ptr<hdl_global_localization::srv::SetGlobalMap::Request> pending_globalmap_req_;

  // 参数
  std::string robot_odom_frame_id_;
  std::string odom_child_frame_id_;
  bool use_imu_;
  bool invert_acc_;
  bool invert_gyro_;
  bool use_global_localization_;

  // IMU数据
  std::mutex imu_data_mutex_;
  std::vector<sensor_msgs::msg::Imu::SharedPtr> imu_data_;

  // 地图和配准
  pcl::PointCloud<PointT>::Ptr globalmap_;
  pcl::Filter<PointT>::Ptr downsample_filter_;
  pcl::Registration<PointT, PointT>::Ptr registration_;

  // 姿态估计器
  std::mutex pose_estimator_mutex_;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator_;

  // 全局定位
  std::atomic_bool relocalizing_;
  std::unique_ptr<DeltaEstimater> delta_estimater_;
  pcl::PointCloud<PointT>::ConstPtr last_scan_;

  // 自动重定位
  std::atomic_bool auto_reloc_triggered_{false};
  std::atomic_int auto_reloc_attempts_{0};
  int auto_reloc_max_attempts_{5};
  rclcpp::TimerBase::SharedPtr auto_reloc_timer_;
};

} // namespace hdl_localization

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto node = std::make_shared<hdl_localization::HdlLocalizationNode>(options);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
