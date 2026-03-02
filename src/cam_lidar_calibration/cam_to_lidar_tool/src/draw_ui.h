
#pragma once
#ifndef DRAW_UI_H
#define DRAW_UI_H

#include "io_yaml.h"
#include "projection.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>

extern std::vector<Eigen::Vector3f> point_cloud_data;
extern std::string cam_instrinsics_path;
extern std::string cam_to_lidar_extrinsics_path;



using namespace calibtool;



namespace calibtool {

    void drawCamToLidarChange(int value, void *param);
}

#endif //LIBCALIBTOOL_DRAW_UI_H
