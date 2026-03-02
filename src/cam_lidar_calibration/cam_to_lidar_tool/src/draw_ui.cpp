
#include <opencv2/opencv.hpp>

#include "draw_ui.h"

std::vector<Eigen::Vector3f> point_cloud_data;
std::string cam_instrinsics_path;
std::string cam_to_lidar_extrinsics_path;

namespace calibtool {
double colmap[50][3] = {{0, 0, 0.5385},           {0, 0, 0.6154},
                        {0, 0, 0.6923},           {0, 0, 0.7692},
                        {0, 0, 0.8462},           {0, 0, 0.9231},
                        {0, 0, 1.0000},           {0, 0.0769, 1.0000},
                        {0, 0.1538, 1.0000},      {0, 0.2308, 1.0000},
                        {0, 0.3846, 1.0000},      {0, 0.4615, 1.0000},
                        {0, 0.5385, 1.0000},      {0, 0.6154, 1.0000},
                        {0, 0.6923, 1.0000},      {0, 0.7692, 1.0000},
                        {0, 0.8462, 1.0000},      {0, 0.9231, 1.0000},
                        {0, 1.0000, 1.0000},      {0.0769, 1.0000, 0.9231},
                        {0.1538, 1.0000, 0.8462}, {0.2308, 1.0000, 0.7692},
                        {0.3077, 1.0000, 0.6923}, {0.3846, 1.0000, 0.6154},
                        {0.4615, 1.0000, 0.5385}, {0.5385, 1.0000, 0.4615},
                        {0.6154, 1.0000, 0.3846}, {0.6923, 1.0000, 0.3077},
                        {0.7692, 1.0000, 0.2308}, {0.8462, 1.0000, 0.1538},
                        {0.9231, 1.0000, 0.0769}, {1.0000, 1.0000, 0},
                        {1.0000, 0.9231, 0},      {1.0000, 0.8462, 0},
                        {1.0000, 0.7692, 0},      {1.0000, 0.6923, 0},
                        {1.0000, 0.6154, 0},      {1.0000, 0.5385, 0},
                        {1.0000, 0.4615, 0},      {1.0000, 0.3846, 0},
                        {1.0000, 0.3077, 0},      {1.0000, 0.2308, 0},
                        {1.0000, 0.1538, 0},      {1.0000, 0.0769, 0},
                        {1.0000, 0, 0},           {0.9231, 0, 0},
                        {0.8462, 0, 0},           {0.7692, 0, 0},
                        {0.6923, 0, 0},           {0.6154, 0, 0}};

void quaterniondTransEuler(Eigen::Quaterniond &q_input, double &yaw,
                           double &roll, double &pitch) {
  Eigen::Quaterniond q = q_input.normalized();
  // Euler is rpy, q是载体系到导航系
  double roll_rad =
      atan2(2 * (q.y() * q.z() + q.w() * q.x()),
            (q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z()));
  double pitch_rad = asin(-2 * q.x() * q.z() + 2 * q.w() * q.y());
  double yaw_rad =
      atan2(2 * (q.x() * q.y() + q.w() * q.z()),
            (q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z()));

  yaw = yaw_rad * 180 / M_PI + 90.0;
  roll = roll_rad * 180 / M_PI + 90.0;
  pitch = pitch_rad * 180 / M_PI;
}

// lidar-camera
void drawCamToLidarProject(cv::Mat &img, Eigen::Vector3d &euler,
                               Eigen::Vector3d &t, std::vector<double> &K,
                               std::vector<double> &D,
                               Eigen::Quaterniond &q_out,
                               Eigen::Vector3d &t_out, float xx_min,
                               float xx_max) {
  for (auto &point : point_cloud_data) {
    double dis_lidar =
        sqrt(pow(point.x(), 2) + pow(point.y(), 2) + pow(point.z(), 2));
    double point_front_x = point.x();

    Eigen::Vector2f uv;
    int range = 0;
    if ((dis_lidar > xx_min) && (dis_lidar < xx_max) && (point_front_x > 0.0)) {
      if (getLidar3dToTheoreticalUV(point, euler, t, K, uv, q_out, t_out,
                                    range) &&
          uv.x() < (float)img.cols && uv.y() < (float)img.rows) {
        int point_size = 2;
        cv::circle(
            img, cv::Point(uv.x(), uv.y()), point_size,
            CV_RGB(255 * colmap[50 - range][0], 255 * colmap[50 - range][1],
                   255 * colmap[50 - range][2]),
            -1);
      }
    }
  }
}

// lidar-around-camera
void drawCamToLidarChange(int value, void *param) {
  std::vector<double> K = {}, D = {};
  Eigen::Quaterniond q0;
  Eigen::Vector3d t0;

  bool flag1 = readExtrinsicsYamlConfig(cam_to_lidar_extrinsics_path, q0, t0);
    if (!flag1) {
      LOG(INFO)
          << "read extrinsics yaml file error. please check path!"
          << "\n";

      exit(-1);
    }
  
  bool flag2 = readIntrinsicsYamlConfig(cam_instrinsics_path, K, D);
  if (!flag2) {
    exit(-1);
  }

  Eigen::Vector3d eulerAngle = q0.normalized().toRotationMatrix().eulerAngles(
      2, 1, 0);  // ZYX, yaw, roll, pitch
  auto yaw_0_deg = (float)(eulerAngle[0] * 180 / M_PI);
  auto roll_0_deg = (float)(eulerAngle[1] * 180 / M_PI);
  auto pitch_0_deg = (float)(eulerAngle[2] * 180 / M_PI);

  float xx_min =
      (float)(cv::getTrackbarPos("X_min", "Cam-To-Lidar-Calibration"));
  float xx_max =
      (float)(cv::getTrackbarPos("X_max", "Cam-To-Lidar-Calibration"));

  float k_x = (-1 - 1.0) / (0 - 200.0);  // 1 cm
  auto b_x = (float)t0(0);
  float xx = (float)cv::getTrackbarPos("X/m", "Cam-To-Lidar-Calibration");
  xx = k_x * (xx - 100) + b_x;

  float k_y = (-1 - 1.0) / (0 - 200.0);
  auto b_y = (float)t0(1);
  float yy = (float)cv::getTrackbarPos("Y/m", "Cam-To-Lidar-Calibration");
  yy = k_y * (yy - 100) + b_y;

  float k_z = (-1 - 1.0) / (0 - 200.0);
  auto b_z = (float)t0(2);
  float zz = (float)cv::getTrackbarPos("Z/m", "Cam-To-Lidar-Calibration");
  zz = k_z * (zz - 100) + b_z;

  float k_pitch = (-5.0 - 5.0) / (0 - 600.0);  // 1'
  float b_pitch = pitch_0_deg;
  float pitch =
      (float)cv::getTrackbarPos("Pitch/°", "Cam-To-Lidar-Calibration");
  pitch = k_pitch * (pitch - 300) + b_pitch;

  float k_roll = (-5.0 - 5.0) / (0 - 600.0);
  float b_roll = roll_0_deg;
  float roll =
      (float)cv::getTrackbarPos("Roll/°", "Cam-To-Lidar-Calibration");
  roll = k_roll * (roll - 300) + b_roll;

  float k_yaw = (-5.0 - 5.0) / (0 - 600.0);
  float b_yaw = yaw_0_deg;
  float yaw = (float)cv::getTrackbarPos("Yaw/°", "Cam-To-Lidar-Calibration");
  yaw = k_yaw * (yaw - 300) + b_yaw;

  cv::Mat src = *(cv::Mat *)param;
  cv::Mat dst;
  dst.release();

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << K[0], K[1], K[2], K[3],
                          K[4], K[5], K[6], K[7], K[8]);

  cv::Mat distCoeffs;
  if (static_cast<int>(D.size()) == 5) {
    distCoeffs =
      (cv::Mat_<double>(5, 1) << D[0], D[1], D[2], D[3], D[4]); 
  }
  if (static_cast<int>(D.size()) == 8) {
    distCoeffs =
      (cv::Mat_<double>(8, 1) << D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7]); 
  }

  LOG(INFO) << "D: " << D[0] << " " << D[1] << " " << D[2]
            << " " << D[3] << " " << D[4];
  undistort(src, dst, cameraMatrix, distCoeffs);
  // dst = src.clone();

  Eigen::Vector3d euler(yaw, roll, pitch);
  // Eigen::Vector3d t(b_x, b_y, b_z);
  Eigen::Vector3d t(xx, yy, zz);
  Eigen::Quaterniond q_out;
  Eigen::Vector3d t_out;
  drawCamToLidarProject(dst, euler, t, K, D, q_out, t_out, xx_min, xx_max);
  std::string out_yaml_path = "out_cam_to_lidar_extrinsics.yaml"; 
  writeExtrinsicsYamlConfig(out_yaml_path, q_out, t_out);
  double yaw_show, roll_show, pitch_show;
  quaterniondTransEuler(q_out, yaw_show, roll_show, pitch_show);

  std::string info_mode = "cam_to_lidar_extrinsics";
  cv::putText(dst, info_mode, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  char info[512];
  sprintf(info, "X:%.2f", t_out.x());
  cv::putText(dst, info, cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  sprintf(info, "Y:%.2f", t_out.y());
  cv::putText(dst, info, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  sprintf(info, "Z:%.2f", t_out.z());
  cv::putText(dst, info, cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  sprintf(info, "Pitch:%.2f", roll_show);
  cv::putText(dst, info, cv::Point(0, 250), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  sprintf(info, "Roll:%.2f", pitch_show);
  cv::putText(dst, info, cv::Point(0, 300), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  sprintf(info, "Yaw:%.2f", yaw_show);
  cv::putText(dst, info, cv::Point(0, 350), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2);
  cv::imwrite("out.jpg", dst);
  if (dst.rows > 2000) {
    cv::resize(dst, dst, cv::Size(int(dst.cols / 3), int(dst.rows / 3)));
  }
  cv::imshow("Cam-To-Lidar-Calibration", dst);
  cv::imwrite("out.jpg", dst);
}


}  // namespace calibtool
