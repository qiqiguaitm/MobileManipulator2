
#include <opencv2/opencv.hpp>

#include "io_yaml.h"

namespace calibtool {

bool readExtrinsicsYamlConfig(const std::string &file_path,
                              Eigen::Quaterniond &q, Eigen::Vector3d &t) {
  YAML::Node config = YAML::LoadFile(file_path);
  if (config.IsNull()) {
    LOG(ERROR) << "Load " << file_path << " failed! please check!";
    return false;
  }
  if (config["transform"]["translation"]) {
    t.x() = config["transform"]["translation"]["x"].as<double>();
    t.y() = config["transform"]["translation"]["y"].as<double>();
    t.z() = config["transform"]["translation"]["z"].as<double>();
  } else {
    LOG(ERROR) << "Read " << file_path << "YAML file failed! please check!";
    return false;
  }
  if (config["transform"]["rotation"]) {
    q.x() = config["transform"]["rotation"]["x"].as<double>();
    q.y() = config["transform"]["rotation"]["y"].as<double>();
    q.z() = config["transform"]["rotation"]["z"].as<double>();
    q.w() = config["transform"]["rotation"]["w"].as<double>();
  } else {
    LOG(ERROR) << "Read " << file_path << "YAML file failed! please check!";
    return false;
  }

  return true;
}

bool readIntrinsicsYamlConfig(const std::string &file_path,
                              std::vector<double> &K, std::vector<double> &D) {
  YAML::Node node = YAML::LoadFile(file_path);
  if (node.IsNull()) {
    LOG(ERROR) << "Load " << file_path << " failed! please check!";
    return false;
  }

  try {
    for (size_t i = 0; i < 9; ++i) {
      K.push_back(node["K"][i].as<double>());
    }
    
    if (static_cast<int>(node["D"].size()) == 5) {
      for (size_t i = 0; i < 5; ++i) {
        D.push_back(node["D"][i].as<double>());
      }
    } else if (static_cast<int>(node["D"].size()) == 8) {
      for (size_t i = 0; i < 8; ++i) {
        D.push_back(node["D"][i].as<double>());
      }
    } else {
      return false;
    }
  } catch (YAML::Exception &e) {
    LOG(ERROR) << "load camera intrisic file " << file_path
               << " with error, YAML exception: " << e.what();
    return false;
  }

  return true;
}

void writeExtrinsicsYamlConfig(const std::string &file_path,
                               Eigen::Quaterniond &q, Eigen::Vector3d &t) {
  std::ofstream outfile;
  outfile.open(file_path);
  std::string hearder;
    hearder =
        "header:\n"
        "  frame_id: lidar\n"
        "child_frame_id: cam\n"
        "transform:\n"
        "  translation:";

  outfile << hearder << "\n";
  outfile << "    x: " << t(0) << "\n";
  outfile << "    y: " << t(1) << "\n";
  outfile << "    z: " << t(2) << "\n";
  outfile << "  rotation: "
          << "\n";
  outfile << "    x: " << q.x() << "\n";
  outfile << "    y: " << q.y() << "\n";
  outfile << "    z: " << q.z() << "\n";
  outfile << "    w: " << q.w() << "\n";

  outfile.close();
}

}  // namespace calibtool
