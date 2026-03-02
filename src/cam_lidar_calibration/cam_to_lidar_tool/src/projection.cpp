
#include <opencv2/opencv.hpp>

#include "projection.h"

namespace calibtool {

bool getLidar3dToTheoreticalUV(const Eigen::Vector3f &xyz,
                               Eigen::Vector3d euler, Eigen::Vector3d &t,
                               std::vector<double> &K, Eigen::Vector2f &uv,
                               Eigen::Quaterniond &q_out,
                               Eigen::Vector3d &t_out, int &range) {
  t_out = t;
  Eigen::Matrix3d r_out;
  r_out = Eigen::AngleAxisd(euler(0) * M_PI / 180, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(euler(1) * M_PI / 180, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(euler(2) * M_PI / 180, Eigen::Vector3d::UnitX());
  Eigen::Quaterniond quaterniond_out(r_out);
  q_out = quaterniond_out.normalized();

  // set the intrinsic and extrinsic matrix
  double matrix1[3][3] = {
      {K[0], K[1], K[2]}, {K[3], K[4], K[5]}, {K[6], K[7], K[8]}};
  double matrix2[4][4] = {{r_out(0, 0), r_out(0, 1), r_out(0, 2), t_out(0)},
                          {r_out(1, 0), r_out(1, 1), r_out(1, 2), t_out(1)},
                          {r_out(2, 0), r_out(2, 1), r_out(2, 2), t_out(2)},
                          {0, 0, 0, 1}};
  double matrix3[4][1] = {xyz.x(), xyz.y(), xyz.z(), 1};

  // transform into the opencv matrix
  cv::Mat matrixIn(3, 3, CV_64F, matrix1);
  cv::Mat matrixOutTemp(4, 4, CV_64F, matrix2);
  cv::Mat matrixOutTemp1 = matrixOutTemp.inv();
  cv::Mat matrixOut = matrixOutTemp1.rowRange(0, 3).clone();
  cv::Mat coordinate(4, 1, CV_64F, matrix3);

  // calculate the result of u and v
  cv::Mat point3d_img = matrixOut * coordinate;
  // double dis = sqrt(pow(point3d_img.at<double>(0, 0), 2) +
  // pow(point3d_img.at<double>(1, 0), 2) + pow(point3d_img.at<double>(2, 0),
  // 2));
  double dis = sqrt(pow(xyz.x(), 2) + pow(xyz.y(), 2) + pow(xyz.z(), 2));
  range = std::min(round((fabs(dis) / 49.0) * 49), 49.0);

  cv::Mat result = matrixIn * point3d_img;

  float u = result.at<double>(0, 0);
  float v = result.at<double>(1, 0);
  float depth = result.at<double>(2, 0);
  // LOG(INFO) << "u-v-depth: " << u << " " << v << " " << depth;
  uv.x() = u / depth;
  uv.y() = v / depth;
  // LOG(INFO) << "uv: " << uv.x() << " " << uv.y();

  return (uv.x() > 0) && (uv.y() > 0);
}


}  // namespace calibtool
