
#include "../src/draw_ui.h"
#include <glog/logging.h>


int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;

    if (argc != 5)
    {
        LOG(ERROR) << "please input: ./camToLidar image_path lidar_path cam_instrinsics_path cam_to_lidar_extrinsics_path" << "\n";
        LOG(ERROR) << "example: ./camToLidar ../data/D455_lidar/1.png ../data/D455_lidar/1.pcd ../paras/D455_intrinsics.yaml ../paras/D455_to_lidar_extrinsics.yaml\n";
        return -1;
    }
    LOG(INFO) << "**camera-lidar calibration tool**" << "\n";
    LOG(INFO) << "How to use:" << "\n";
    LOG(INFO)
        << "1) use the mouse to click the sliding window and change the value of XYZ-RPY, x-min-max distance and 'Mode';"
        << "\n";
    LOG(INFO)
        << "2) confirm the new extrinsics by compare the coincidence of the point cloud's projection and outline on the image;"
        << "\n";
    LOG(INFO) << "3) end the UI by click the 'SPACE' key of the keyboard;" << "\n";
    LOG(INFO) << "4) restart the UI by click the red 'x' on the UI's up-right;" << "\n";
    LOG(INFO) << "5) the new yaml file of extrinsics will be saved under the 'build' path." << "\n";
    LOG(INFO) << ".." << "\n";
    cv::Mat src = cv::imread(argv[1]);
    std::string lidar_path = argv[2];
    cam_instrinsics_path = argv[3];
    cam_to_lidar_extrinsics_path = argv[4];

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(lidar_path, *cloud) == -1)
    {
        LOG(ERROR) << "read pcd file failed" << "\n";
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZI> pcd = *cloud;
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    for (auto &point : cloud->points)
    {
        if ((point.x > -100) && (point.x < 200) && (point.y < 60) && (point.y > -60))
        {
            point_cloud_data.emplace_back(point.x, point.y, point.z);
        }
    }
    LOG(INFO) << "point_cloud_data size: " << point_cloud_data.size() << "\n";

    bool flag = true;
    while (flag)
    {
        cv::namedWindow("Cam-To-Lidar-Calibration", cv::WINDOW_AUTOSIZE); // cv::WINDOW_AUTOSIZE

        int X_min = 0;
        cv::createTrackbar("X_min", "Cam-To-Lidar-Calibration", &X_min, 200, calibtool::drawCamToLidarChange, &src);
        int X_max = 30;
        cv::createTrackbar("X_max", "Cam-To-Lidar-Calibration", &X_max, 200, calibtool::drawCamToLidarChange, &src);

        int X = 100;
        cv::createTrackbar("X/m", "Cam-To-Lidar-Calibration", &X, 200, calibtool::drawCamToLidarChange, &src);

        int Y = 100;
        cv::createTrackbar("Y/m", "Cam-To-Lidar-Calibration", &Y, 200, calibtool::drawCamToLidarChange, &src);

        int Z = 100;
        cv::createTrackbar("Z/m", "Cam-To-Lidar-Calibration", &Z, 200, calibtool::drawCamToLidarChange, &src);

        int pitch = 300;
        cv::createTrackbar("Pitch/°", "Cam-To-Lidar-Calibration", &pitch, 600, calibtool::drawCamToLidarChange, &src);

        int roll = 300;
        cv::createTrackbar("Roll/°", "Cam-To-Lidar-Calibration", &roll, 600, calibtool::drawCamToLidarChange, &src);
        int yaw = 300;
        cv::createTrackbar("Yaw/°", "Cam-To-Lidar-Calibration", &yaw, 600, calibtool::drawCamToLidarChange, &src);
        calibtool::drawCamToLidarChange(0, &src);

        char chKey = cv::waitKey();
        if (chKey == 32)
        { // SPACE key
            flag = false;
        }
    }

    LOG(INFO) << "**finished**" << "\n";
    LOG(INFO) << "**output:** " << "\n";
    LOG(INFO) << " 1) out.jpg;" << "\n";
    LOG(INFO) << " 2) out_cam_to_lidar_extrinsics.yaml." << "\n";
    LOG(INFO) << "****" << "\n";

    google::ShutdownGoogleLogging();

    return 0;
}
