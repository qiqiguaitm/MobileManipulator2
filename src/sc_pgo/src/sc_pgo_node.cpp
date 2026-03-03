/**
 * SC-PGO Node for ROS2
 * Scan Context Pose Graph Optimization
 *
 * Migrated from ROS1 laserPosegraphOptimization.cpp
 * Original: https://github.com/gisbi-kim/SC-A-LOAM
 */

#include <fstream>
#include <cmath>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <iomanip>
#include <chrono>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Eigen
#include <Eigen/Dense>

// GTSAM
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

// Local
#include "sc_pgo/common.h"
#include "sc_pgo/tic_toc.h"
#include "sc_pgo/Scancontext.h"

using namespace gtsam;

// Pose6D结构已在common.h中定义

class ScPgoNode : public rclcpp::Node {
public:
    ScPgoNode() : Node("sc_pgo") {
        // 声明参数
        declare_parameter("save_directory", "/home/didi/workspace/MobileManipulator2/maps/sc_pgo/");
        declare_parameter("keyframe_meter_gap", 0.5);
        declare_parameter("keyframe_deg_gap", 10.0);
        declare_parameter("sc_dist_thres", 0.15);
        declare_parameter("sc_max_radius", 20.0);
        declare_parameter("mapviz_filter_size", 0.05);
        declare_parameter("icp_fitness_threshold", 0.3);

        // 获取参数
        std::string base_save_dir = get_parameter("save_directory").as_string();
        keyframe_meter_gap_ = get_parameter("keyframe_meter_gap").as_double();
        keyframe_deg_gap_ = get_parameter("keyframe_deg_gap").as_double();
        double sc_dist_thres = get_parameter("sc_dist_thres").as_double();
        double sc_max_radius = get_parameter("sc_max_radius").as_double();
        double mapviz_filter_size = get_parameter("mapviz_filter_size").as_double();

        keyframe_rad_gap_ = keyframe_deg_gap_ * M_PI / 180.0;

        // 在 save_directory 下创建时间戳子目录，每次建图数据独立保存
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&t, &tm_buf);
        char ts_buf[32];
        std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d_%H%M%S", &tm_buf);

        // 确保 base 目录以 / 结尾
        if (!base_save_dir.empty() && base_save_dir.back() != '/') {
            base_save_dir += '/';
        }
        save_directory_ = base_save_dir + std::string(ts_buf) + "/";

        // 初始化路径
        pg_kitti_format_ = save_directory_ + "optimized_poses.txt";
        odom_kitti_format_ = save_directory_ + "odom_poses.txt";
        pg_scans_directory_ = save_directory_ + "Scans/";

        // 创建目录
        std::system(("mkdir -p " + pg_scans_directory_).c_str());

        // 创建 latest 软链接指向本次建图目录
        std::string latest_link = base_save_dir + "latest";
        std::system(("rm -f " + latest_link).c_str());
        std::system(("ln -s " + save_directory_ + " " + latest_link).c_str());

        // 打开文件
        std::string times_path = save_directory_ + "times.txt";
        pg_time_save_stream_.open(times_path, std::fstream::out);
        pg_time_save_stream_.precision(std::numeric_limits<double>::max_digits10);

        // 初始化ISAM2
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam_ = new gtsam::ISAM2(parameters);
        initNoises();

        // 初始化Scan Context
        sc_manager_.setSCdistThres(sc_dist_thres);
        sc_manager_.setMaximumRadius(sc_max_radius);

        // 初始化降采样器
        float filter_size = 0.4f;
        down_size_filter_sc_.setLeafSize(filter_size, filter_size, filter_size);
        down_size_filter_icp_.setLeafSize(filter_size, filter_size, filter_size);
        down_size_filter_map_pgo_.setLeafSize(mapviz_filter_size, mapviz_filter_size, mapviz_filter_size);

        // 创建订阅者
        sub_laser_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered_body", 100,
            std::bind(&ScPgoNode::laserCloudHandler, this, std::placeholders::_1));

        sub_laser_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry", 100,
            std::bind(&ScPgoNode::laserOdometryHandler, this, std::placeholders::_1));

        sub_gps_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/gps/fix", 100,
            std::bind(&ScPgoNode::gpsHandler, this, std::placeholders::_1));

        // 创建发布者
        pub_odom_aft_pgo_ = create_publisher<nav_msgs::msg::Odometry>("/aft_pgo_odom", 100);
        pub_path_aft_pgo_ = create_publisher<nav_msgs::msg::Path>("/aft_pgo_path", 100);
        pub_map_aft_pgo_ = create_publisher<sensor_msgs::msg::PointCloud2>("/aft_pgo_map", 100);
        pub_loop_scan_local_ = create_publisher<sensor_msgs::msg::PointCloud2>("/loop_scan_local", 100);
        pub_loop_submap_local_ = create_publisher<sensor_msgs::msg::PointCloud2>("/loop_submap_local", 100);

        // TF广播器
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(get_logger(), "SC-PGO Node initialized");
        RCLCPP_INFO(get_logger(), "  Save directory: %s", save_directory_.c_str());
        RCLCPP_INFO(get_logger(), "  Keyframe gap: %.2fm / %.1fdeg", keyframe_meter_gap_, keyframe_deg_gap_);

        // 启动工作线程
        thread_pg_ = std::thread(&ScPgoNode::processPG, this);
        thread_lcd_ = std::thread(&ScPgoNode::processLCD, this);
        thread_icp_ = std::thread(&ScPgoNode::processICP, this);
        thread_isam_ = std::thread(&ScPgoNode::processISAM, this);
        thread_viz_map_ = std::thread(&ScPgoNode::processVizMap, this);
        thread_viz_path_ = std::thread(&ScPgoNode::processVizPath, this);
    }

    ~ScPgoNode() {
        running_ = false;
        if (thread_pg_.joinable()) thread_pg_.join();
        if (thread_lcd_.joinable()) thread_lcd_.join();
        if (thread_icp_.joinable()) thread_icp_.join();
        if (thread_isam_.joinable()) thread_isam_.join();
        if (thread_viz_map_.joinable()) thread_viz_map_.join();
        if (thread_viz_path_.joinable()) thread_viz_path_.join();

        // 确保最终优化结果被保存
        if (gtsam_graph_made_ && !keyframe_poses_updated_.empty()) {
            RCLCPP_INFO(get_logger(), "Saving final optimized poses (%zu keyframes)...", keyframe_poses_updated_.size());
            saveOptimizedPoses();
        }

        if (isam_) delete isam_;
        pg_time_save_stream_.close();
    }

private:
    // 回调函数
    void laserCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_buf_);
        cloud_buf_.push(msg);
    }

    void laserOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_buf_);
        odom_buf_.push(msg);
    }

    void gpsHandler(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        if (use_gps_) {
            std::lock_guard<std::mutex> lock(mtx_buf_);
            gps_buf_.push(msg);
        }
    }

    // 初始化噪声模型
    void initNoises() {
        gtsam::Vector prior_noise_vec(6);
        prior_noise_vec << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        prior_noise_ = noiseModel::Diagonal::Variances(prior_noise_vec);

        gtsam::Vector odom_noise_vec(6);
        odom_noise_vec << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        odom_noise_ = noiseModel::Diagonal::Variances(odom_noise_vec);

        double loop_noise_score = 0.5;
        gtsam::Vector robust_noise_vec(6);
        robust_noise_vec << loop_noise_score, loop_noise_score, loop_noise_score,
                           loop_noise_score, loop_noise_score, loop_noise_score;
        robust_loop_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robust_noise_vec));

        double big_noise = 1000000000.0;
        double gps_altitude_noise = 250.0;
        gtsam::Vector gps_noise_vec(3);
        gps_noise_vec << big_noise, big_noise, gps_altitude_noise;
        robust_gps_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(gps_noise_vec));
    }

    // 从Odometry获取Pose6D
    Pose6D getOdom(const nav_msgs::msg::Odometry::SharedPtr& odom) {
        Pose6D pose;
        pose.x = odom->pose.pose.position.x;
        pose.y = odom->pose.pose.position.y;
        pose.z = odom->pose.pose.position.z;

        tf2::Quaternion q(
            odom->pose.pose.orientation.x,
            odom->pose.pose.orientation.y,
            odom->pose.pose.orientation.z,
            odom->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        m.getRPY(pose.roll, pose.pitch, pose.yaw);

        return pose;
    }

    // Pose6D转GTSAM Pose3
    gtsam::Pose3 pose6DtoGTSAMPose3(const Pose6D& p) {
        return gtsam::Pose3(
            gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw),
            gtsam::Point3(p.x, p.y, p.z));
    }

    // 计算两个Pose6D的差异
    Pose6D diffTransformation(const Pose6D& p1, const Pose6D& p2) {
        Eigen::Affine3f t1 = pcl::getTransformation(p1.x, p1.y, p1.z, p1.roll, p1.pitch, p1.yaw);
        Eigen::Affine3f t2 = pcl::getTransformation(p2.x, p2.y, p2.z, p2.roll, p2.pitch, p2.yaw);
        Eigen::Matrix4f delta_mat = t1.matrix().inverse() * t2.matrix();
        Eigen::Affine3f delta;
        delta.matrix() = delta_mat;

        float dx, dy, dz, droll, dpitch, dyaw;
        pcl::getTranslationAndEulerAngles(delta, dx, dy, dz, droll, dpitch, dyaw);

        return Pose6D{std::abs(dx), std::abs(dy), std::abs(dz),
                      std::abs(droll), std::abs(dpitch), std::abs(dyaw)};
    }

    // 点云从局部坐标系转换到全局坐标系
    pcl::PointCloud<PointType>::Ptr local2global(
        const pcl::PointCloud<PointType>::Ptr& cloud_in, const Pose6D& tf) {

        pcl::PointCloud<PointType>::Ptr cloud_out(new pcl::PointCloud<PointType>());
        int cloud_size = cloud_in->size();
        cloud_out->resize(cloud_size);

        Eigen::Affine3f trans = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < cloud_size; ++i) {
            const auto& pt_from = cloud_in->points[i];
            cloud_out->points[i].x = trans(0,0)*pt_from.x + trans(0,1)*pt_from.y + trans(0,2)*pt_from.z + trans(0,3);
            cloud_out->points[i].y = trans(1,0)*pt_from.x + trans(1,1)*pt_from.y + trans(1,2)*pt_from.z + trans(1,3);
            cloud_out->points[i].z = trans(2,0)*pt_from.x + trans(2,1)*pt_from.y + trans(2,2)*pt_from.z + trans(2,3);
            cloud_out->points[i].intensity = pt_from.intensity;
        }

        return cloud_out;
    }

    // 运行ISAM2优化
    void runISAM2opt() {
        isam_->update(gtsam_graph_, initial_estimate_);
        isam_->update();

        gtsam_graph_.resize(0);
        initial_estimate_.clear();

        isam_current_estimate_ = isam_->calculateEstimate();
        updatePoses();
    }

    // 更新优化后的位姿
    void updatePoses() {
        std::lock_guard<std::mutex> lock(mtx_kf_);
        for (int i = 0; i < (int)isam_current_estimate_.size(); i++) {
            Pose6D& p = keyframe_poses_updated_[i];
            auto pose = isam_current_estimate_.at<gtsam::Pose3>(i);
            p.x = pose.translation().x();
            p.y = pose.translation().y();
            p.z = pose.translation().z();
            p.roll = pose.rotation().roll();
            p.pitch = pose.rotation().pitch();
            p.yaw = pose.rotation().yaw();
        }

        std::lock_guard<std::mutex> lock2(mtx_recent_pose_);
        if (!isam_current_estimate_.empty()) {
            auto last_pose = isam_current_estimate_.at<gtsam::Pose3>(isam_current_estimate_.size()-1);
            recent_optimized_x_ = last_pose.translation().x();
            recent_optimized_y_ = last_pose.translation().y();
        }
        recent_idx_updated_ = keyframe_poses_updated_.size() - 1;
    }

    // 获取回环关键帧附近的点云
    void loopFindNearKeyframesCloud(
        pcl::PointCloud<PointType>::Ptr& near_keyframes,
        int key, int submap_size, int root_idx) {

        near_keyframes->clear();
        for (int i = -submap_size; i <= submap_size; ++i) {
            int key_near = key + i;
            if (key_near < 0 || key_near >= (int)keyframe_clouds_.size())
                continue;

            std::lock_guard<std::mutex> lock(mtx_kf_);
            *near_keyframes += *local2global(keyframe_clouds_[key_near], keyframe_poses_updated_[root_idx]);
        }

        if (near_keyframes->empty()) return;

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        down_size_filter_icp_.setInputCloud(near_keyframes);
        down_size_filter_icp_.filter(*cloud_temp);
        *near_keyframes = *cloud_temp;
    }

    // ICP回环验证
    std::optional<gtsam::Pose3> doICPVirtualRelative(int loop_kf_idx, int curr_kf_idx) {
        int history_search_num = 25;
        pcl::PointCloud<PointType>::Ptr curr_cloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr target_cloud(new pcl::PointCloud<PointType>());

        loopFindNearKeyframesCloud(curr_cloud, curr_kf_idx, 0, loop_kf_idx);
        loopFindNearKeyframesCloud(target_cloud, loop_kf_idx, history_search_num, loop_kf_idx);

        // 发布用于调试
        sensor_msgs::msg::PointCloud2 curr_msg, target_msg;
        pcl::toROSMsg(*curr_cloud, curr_msg);
        curr_msg.header.frame_id = "camera_init";
        curr_msg.header.stamp = now();
        pub_loop_scan_local_->publish(curr_msg);

        pcl::toROSMsg(*target_cloud, target_msg);
        target_msg.header.frame_id = "camera_init";
        target_msg.header.stamp = now();
        pub_loop_submap_local_->publish(target_msg);

        // ICP配准 (室内环境优化参数)
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2.0);  // 室内环境2m合理值 (原150m过大)
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(100);  // 启用RANSAC过滤离群点 (原0=禁用)

        icp.setInputSource(curr_cloud);
        icp.setInputTarget(target_cloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        float fitness_threshold = get_parameter("icp_fitness_threshold").as_double();
        if (!icp.hasConverged() || icp.getFitnessScore() > fitness_threshold) {
            RCLCPP_INFO(get_logger(), "[SC loop] ICP failed (%.3f > %.3f)",
                       icp.getFitnessScore(), fitness_threshold);
            return std::nullopt;
        }

        RCLCPP_INFO(get_logger(), "[SC loop] ICP passed (%.3f < %.3f)",
                   icp.getFitnessScore(), fitness_threshold);

        // 获取变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correction;
        correction = icp.getFinalTransformation();
        pcl::getTranslationAndEulerAngles(correction, x, y, z, roll, pitch, yaw);

        gtsam::Pose3 pose_from = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        gtsam::Pose3 pose_to = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        return pose_from.between(pose_to);
    }

    // 位姿图构建线程
    void processPG() {
        while (running_ && rclcpp::ok()) {
            while (!odom_buf_.empty() && !cloud_buf_.empty()) {
                mtx_buf_.lock();

                // 时间同步
                while (!odom_buf_.empty() &&
                       rclcpp::Time(odom_buf_.front()->header.stamp).seconds() <
                       rclcpp::Time(cloud_buf_.front()->header.stamp).seconds()) {
                    odom_buf_.pop();
                }

                if (odom_buf_.empty()) {
                    mtx_buf_.unlock();
                    break;
                }

                double time_laser_odom = rclcpp::Time(odom_buf_.front()->header.stamp).seconds();
                double time_laser = rclcpp::Time(cloud_buf_.front()->header.stamp).seconds();

                // 获取点云
                pcl::PointCloud<PointType>::Ptr this_keyframe(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(*cloud_buf_.front(), *this_keyframe);
                cloud_buf_.pop();

                // 获取位姿
                Pose6D pose_curr = getOdom(odom_buf_.front());
                odom_buf_.pop();

                // GPS处理(可选)
                double eps = 0.1;
                while (!gps_buf_.empty()) {
                    auto this_gps = gps_buf_.front();
                    double gps_time = rclcpp::Time(this_gps->header.stamp).seconds();
                    if (std::abs(gps_time - time_laser_odom) < eps) {
                        curr_gps_ = this_gps;
                        has_gps_for_this_kf_ = true;
                        break;
                    } else {
                        has_gps_for_this_kf_ = false;
                    }
                    gps_buf_.pop();
                }
                mtx_buf_.unlock();

                // 检查是否为关键帧
                odom_pose_prev_ = odom_pose_curr_;
                odom_pose_curr_ = pose_curr;
                Pose6D dtf = diffTransformation(odom_pose_prev_, odom_pose_curr_);

                double delta_trans = std::sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z);
                translation_accumulated_ += delta_trans;
                rotation_accumulated_ += (dtf.roll + dtf.pitch + dtf.yaw);

                bool is_keyframe = false;
                if (translation_accumulated_ > keyframe_meter_gap_ ||
                    rotation_accumulated_ > keyframe_rad_gap_) {
                    is_keyframe = true;
                    translation_accumulated_ = 0.0;
                    rotation_accumulated_ = 0.0;
                }

                if (!is_keyframe) continue;

                // GPS偏移初始化
                if (!gps_offset_initialized_ && has_gps_for_this_kf_) {
                    gps_altitude_init_offset_ = curr_gps_->altitude;
                    gps_offset_initialized_ = true;
                }

                // 降采样关键帧
                pcl::PointCloud<PointType>::Ptr this_keyframe_ds(new pcl::PointCloud<PointType>());
                down_size_filter_sc_.setInputCloud(this_keyframe);
                down_size_filter_sc_.filter(*this_keyframe_ds);

                // 保存关键帧
                mtx_kf_.lock();
                keyframe_clouds_.push_back(this_keyframe_ds);
                keyframe_poses_.push_back(pose_curr);
                keyframe_poses_updated_.push_back(pose_curr);
                keyframe_times_.push_back(time_laser_odom);

                sc_manager_.makeAndSaveScancontextAndKeys(*this_keyframe_ds);
                mtx_kf_.unlock();

                // 添加因子
                int prev_idx = keyframe_poses_.size() - 2;
                int curr_idx = keyframe_poses_.size() - 1;

                if (!gtsam_graph_made_) {
                    // 第一帧: 添加先验因子
                    gtsam::Pose3 pose_origin = pose6DtoGTSAMPose3(keyframe_poses_[0]);

                    mtx_posegraph_.lock();
                    gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose_origin, prior_noise_));
                    initial_estimate_.insert(0, pose_origin);
                    mtx_posegraph_.unlock();

                    gtsam_graph_made_ = true;
                    RCLCPP_INFO(get_logger(), "PoseGraph prior node 0 added");
                } else {
                    // 添加里程计因子
                    gtsam::Pose3 pose_from = pose6DtoGTSAMPose3(keyframe_poses_[prev_idx]);
                    gtsam::Pose3 pose_to = pose6DtoGTSAMPose3(keyframe_poses_[curr_idx]);

                    mtx_posegraph_.lock();
                    gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        prev_idx, curr_idx, pose_from.between(pose_to), odom_noise_));

                    // GPS因子(可选)
                    if (has_gps_for_this_kf_) {
                        double alt_offset = curr_gps_->altitude - gps_altitude_init_offset_;
                        mtx_recent_pose_.lock();
                        gtsam::Point3 gps_constraint(recent_optimized_x_, recent_optimized_y_, alt_offset);
                        mtx_recent_pose_.unlock();
                        gtsam_graph_.add(gtsam::GPSFactor(curr_idx, gps_constraint, robust_gps_noise_));
                    }

                    initial_estimate_.insert(curr_idx, pose_to);
                    mtx_posegraph_.unlock();

                    if (curr_idx % 50 == 0) {
                        RCLCPP_INFO(get_logger(), "PoseGraph odom node %d added", curr_idx);
                    }
                }

                // 保存点云
                std::ostringstream ss;
                ss << std::setfill('0') << std::setw(6) << curr_idx;
                pcl::io::savePCDFileBinary(pg_scans_directory_ + ss.str() + ".pcd", *this_keyframe);
                pg_time_save_stream_ << time_laser << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // 回环检测线程
    void processLCD() {
        float loop_closure_freq = 1.0;
        rclcpp::Rate rate(loop_closure_freq);

        while (running_ && rclcpp::ok()) {
            rate.sleep();

            if ((int)keyframe_poses_.size() < sc_manager_.NUM_EXCLUDE_RECENT)
                continue;

            auto result = sc_manager_.detectLoopClosureID();
            int loop_kf_idx = result.first;

            if (loop_kf_idx != -1) {
                int curr_kf_idx = keyframe_poses_.size() - 1;
                RCLCPP_INFO(get_logger(), "Loop detected! %d <-> %d", loop_kf_idx, curr_kf_idx);

                std::lock_guard<std::mutex> lock(mtx_buf_);
                sc_loop_icp_buf_.push(std::make_pair(loop_kf_idx, curr_kf_idx));
            }
        }
    }

    // ICP处理线程
    void processICP() {
        while (running_ && rclcpp::ok()) {
            while (!sc_loop_icp_buf_.empty()) {
                if (sc_loop_icp_buf_.size() > 30) {
                    RCLCPP_WARN(get_logger(), "Too many loop candidates waiting: %zu", sc_loop_icp_buf_.size());
                }

                mtx_buf_.lock();
                auto loop_pair = sc_loop_icp_buf_.front();
                sc_loop_icp_buf_.pop();
                mtx_buf_.unlock();

                int prev_idx = loop_pair.first;
                int curr_idx = loop_pair.second;

                auto relative_pose = doICPVirtualRelative(prev_idx, curr_idx);
                if (relative_pose) {
                    mtx_posegraph_.lock();
                    gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        prev_idx, curr_idx, relative_pose.value(), robust_loop_noise_));
                    mtx_posegraph_.unlock();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // ISAM2优化线程
    void processISAM() {
        float isam_freq = 1.0;
        rclcpp::Rate rate(isam_freq);

        while (running_ && rclcpp::ok()) {
            rate.sleep();

            if (gtsam_graph_made_) {
                mtx_posegraph_.lock();
                runISAM2opt();
                RCLCPP_DEBUG(get_logger(), "Running iSAM2 optimization...");
                mtx_posegraph_.unlock();

                // 保存优化结果
                saveOptimizedPoses();
            }
        }
    }

    // 保存优化后的位姿 (写入前备份旧文件)
    void saveOptimizedPoses() {
        // 备份已有文件，避免截断覆盖丢失数据
        if (std::ifstream(pg_kitti_format_).good()) {
            std::string bak = pg_kitti_format_ + ".bak";
            std::rename(pg_kitti_format_.c_str(), bak.c_str());
        }
        std::fstream stream(pg_kitti_format_, std::fstream::out);
        for (const auto& pose : keyframe_poses_updated_) {
            gtsam::Pose3 p = pose6DtoGTSAMPose3(pose);
            auto t = p.translation();
            auto R = p.rotation();
            auto c1 = R.column(1);
            auto c2 = R.column(2);
            auto c3 = R.column(3);

            stream << c1.x() << " " << c2.x() << " " << c3.x() << " " << t.x() << " "
                   << c1.y() << " " << c2.y() << " " << c3.y() << " " << t.y() << " "
                   << c1.z() << " " << c2.z() << " " << c3.z() << " " << t.z() << std::endl;
        }
    }

    // 地图可视化线程
    void processVizMap() {
        float viz_freq = 0.1;  // 10秒一次
        rclcpp::Rate rate(viz_freq);

        while (running_ && rclcpp::ok()) {
            rate.sleep();

            if (recent_idx_updated_ > 1) {
                pubMap();
            }
        }
    }

    // 发布全局地图
    void pubMap() {
        int skip_frames = 2;
        int counter = 0;

        pcl::PointCloud<PointType>::Ptr cloud_map_pgo(new pcl::PointCloud<PointType>());

        mtx_kf_.lock();
        for (int i = 0; i < recent_idx_updated_; i++) {
            if (counter % skip_frames == 0) {
                *cloud_map_pgo += *local2global(keyframe_clouds_[i], keyframe_poses_updated_[i]);
            }
            counter++;
        }
        mtx_kf_.unlock();

        down_size_filter_map_pgo_.setInputCloud(cloud_map_pgo);
        down_size_filter_map_pgo_.filter(*cloud_map_pgo);

        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud_map_pgo, cloud_msg);
        cloud_msg.header.frame_id = "camera_init";
        cloud_msg.header.stamp = now();
        pub_map_aft_pgo_->publish(cloud_msg);
    }

    // 路径可视化线程
    void processVizPath() {
        float viz_freq = 10.0;
        rclcpp::Rate rate(viz_freq);

        while (running_ && rclcpp::ok()) {
            rate.sleep();

            if (recent_idx_updated_ > 1) {
                pubPath();
            }
        }
    }

    // 发布路径
    void pubPath() {
        nav_msgs::msg::Odometry odom_aft_pgo;
        nav_msgs::msg::Path path_aft_pgo;
        path_aft_pgo.header.frame_id = "camera_init";

        mtx_kf_.lock();
        for (int i = 0; i < recent_idx_updated_; i++) {
            const Pose6D& pose = keyframe_poses_updated_[i];

            nav_msgs::msg::Odometry odom_this;
            odom_this.header.frame_id = "camera_init";
            odom_this.child_frame_id = "aft_pgo";
            odom_this.header.stamp = rclcpp::Time(static_cast<int64_t>(keyframe_times_[i] * 1e9));
            odom_this.pose.pose.position.x = pose.x;
            odom_this.pose.pose.position.y = pose.y;
            odom_this.pose.pose.position.z = pose.z;

            tf2::Quaternion q;
            q.setRPY(pose.roll, pose.pitch, pose.yaw);
            odom_this.pose.pose.orientation.x = q.x();
            odom_this.pose.pose.orientation.y = q.y();
            odom_this.pose.pose.orientation.z = q.z();
            odom_this.pose.pose.orientation.w = q.w();

            odom_aft_pgo = odom_this;

            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = odom_this.header;
            pose_stamped.pose = odom_this.pose.pose;
            path_aft_pgo.poses.push_back(pose_stamped);
        }
        mtx_kf_.unlock();

        path_aft_pgo.header.stamp = odom_aft_pgo.header.stamp;
        pub_odom_aft_pgo_->publish(odom_aft_pgo);
        pub_path_aft_pgo_->publish(path_aft_pgo);

        // 发布TF
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = odom_aft_pgo.header.stamp;
        tf.header.frame_id = "camera_init";
        tf.child_frame_id = "aft_pgo";
        tf.transform.translation.x = odom_aft_pgo.pose.pose.position.x;
        tf.transform.translation.y = odom_aft_pgo.pose.pose.position.y;
        tf.transform.translation.z = odom_aft_pgo.pose.pose.position.z;
        tf.transform.rotation = odom_aft_pgo.pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

private:
    // 参数
    std::string save_directory_;
    double keyframe_meter_gap_;
    double keyframe_deg_gap_;
    double keyframe_rad_gap_;
    std::string pg_kitti_format_;
    std::string odom_kitti_format_;
    std::string pg_scans_directory_;
    std::fstream pg_time_save_stream_;

    // 状态
    bool running_ = true;
    bool use_gps_ = false;
    bool gtsam_graph_made_ = false;
    bool gps_offset_initialized_ = false;
    bool has_gps_for_this_kf_ = false;
    double gps_altitude_init_offset_ = 0.0;
    double translation_accumulated_ = 1000000.0;
    double rotation_accumulated_ = 1000000.0;
    double recent_optimized_x_ = 0.0;
    double recent_optimized_y_ = 0.0;
    int recent_idx_updated_ = 0;

    Pose6D odom_pose_prev_{0,0,0,0,0,0};
    Pose6D odom_pose_curr_{0,0,0,0,0,0};

    // 缓冲区
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_buf_;
    std::queue<nav_msgs::msg::Odometry::SharedPtr> odom_buf_;
    std::queue<sensor_msgs::msg::NavSatFix::SharedPtr> gps_buf_;
    std::queue<std::pair<int, int>> sc_loop_icp_buf_;
    sensor_msgs::msg::NavSatFix::SharedPtr curr_gps_;

    // 关键帧数据
    std::vector<pcl::PointCloud<PointType>::Ptr> keyframe_clouds_;
    std::vector<Pose6D> keyframe_poses_;
    std::vector<Pose6D> keyframe_poses_updated_;
    std::vector<double> keyframe_times_;

    // GTSAM
    gtsam::NonlinearFactorGraph gtsam_graph_;
    gtsam::Values initial_estimate_;
    gtsam::ISAM2* isam_ = nullptr;
    gtsam::Values isam_current_estimate_;
    noiseModel::Diagonal::shared_ptr prior_noise_;
    noiseModel::Diagonal::shared_ptr odom_noise_;
    noiseModel::Base::shared_ptr robust_loop_noise_;
    noiseModel::Base::shared_ptr robust_gps_noise_;

    // Scan Context
    SCManager sc_manager_;

    // 滤波器
    pcl::VoxelGrid<PointType> down_size_filter_sc_;
    pcl::VoxelGrid<PointType> down_size_filter_icp_;
    pcl::VoxelGrid<PointType> down_size_filter_map_pgo_;

    // 互斥锁
    std::mutex mtx_buf_;
    std::mutex mtx_kf_;
    std::mutex mtx_posegraph_;
    std::mutex mtx_recent_pose_;

    // 线程
    std::thread thread_pg_;
    std::thread thread_lcd_;
    std::thread thread_icp_;
    std::thread thread_isam_;
    std::thread thread_viz_map_;
    std::thread thread_viz_path_;

    // ROS2订阅者
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_laser_cloud_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_laser_odom_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gps_;

    // ROS2发布者
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_aft_pgo_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_aft_pgo_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_aft_pgo_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_loop_scan_local_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_loop_submap_local_;

    // TF广播器
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ScPgoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
