# cam_to_lidar tool

动态手动调整标定参数可视化小工具

1. camToLidar 标定camera-lidar外参数；


## Bulid and use

```
mkdir build
cd build
cmake ..
make
./camToLidar image_path lidar_path cam_instrinsics_path cam_to_lidar_extrinsics_path
```

- use

``` 
./camToLidar ../data/D455_lidar/1.png ../data/D455_lidar/1.pcd ../paras/D455_intrinsics.yaml ../paras/D455_to_lidar_extrinsics.yaml

```


## Requirement

- c++14
- opencv 3
- pcl 1.10
- eigen3
- yaml-cpp
  

## Sample use


### camera_to_lidar

camera_to_lidar.cpp

其中：

//drawCamToLidarChange函数为相机-激光雷达外参数动态调整主函数

- input: 

```
image_path: 输入图片路径; 
lidar_path: 输入激光雷达路径; 
cam_instrinsics_path: 输入相机内参数路径；
cam_to_lidar_extrinsics_path: 输入相机与激光雷达默认初始外参路径；
x_min: 雷达点云最近距离 范围0-200 m, 初始值为0 m;
x_max: 雷达点云最远距离 范围0-200 m,初始值为30 m;
```
     
- output: 
```
out.jpg: 更新外参数后点云投影效果图,build路径下.
extrinsics yaml file: 更新后外参数yaml文件, build路径下.
```



## 界面操作说明
```
1）使用鼠标左右滑动滑动窗，修改滑动窗的值；
2）通过观察几何特征或投影轮廓重合效果确定新外参值；
3）键盘空格键结束调参；
4）鼠标点击图像关闭按钮可以重置当前外参，重新进行外参调整；
5）新外参yaml文件保存在build路径下。
```

- 滑动窗精度和调整范围：
```
X-Y-Z：单位m，调整范围-1m ~ 1m，精度1cm；
Roll-Pitch-Yaw：单位度，调整范围-5度 ~ 5度，精度1分(1/60度)。
```

