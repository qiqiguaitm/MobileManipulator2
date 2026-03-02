
#pragma once
#ifndef IO_YAML_H
#define IO_YAML_H
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>


namespace calibtool {

bool readExtrinsicsYamlConfig(const std::string &file_path,
                              Eigen::Quaterniond &q, Eigen::Vector3d &t);


bool readIntrinsicsYamlConfig(const std::string &file_path,
                              std::vector<double> &K, std::vector<double> &D);


void writeExtrinsicsYamlConfig(const std::string &file_path,
                               Eigen::Quaterniond &q, Eigen::Vector3d &t);

}  // namespace calibtool

#endif  // LIBCALIBTOOL_IO_YAML_H
