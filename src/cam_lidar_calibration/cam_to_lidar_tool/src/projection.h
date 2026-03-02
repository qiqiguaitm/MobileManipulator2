
#pragma once
#ifndef PROJECTION_H
#define PROJECTION_H
#include "io_yaml.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

using namespace calibtool;

namespace calibtool
{ 
    bool getLidar3dToTheoreticalUV(const Eigen::Vector3f &xyz, Eigen::Vector3d euler, Eigen::Vector3d &t, std::vector<double> &K,
                                   Eigen::Vector2f &uv, Eigen::Quaterniond &q_out, Eigen::Vector3d &t_out, int &range);

}

#endif // LIBCALIBTOOL_PROJECTION_H
