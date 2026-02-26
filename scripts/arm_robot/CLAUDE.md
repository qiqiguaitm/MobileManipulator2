# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a robotic grasping system that integrates computer vision, object detection, and robotic arm control. The system supports multiple robotic platforms (XArm and Piper) with interchangeable interfaces.

### Core System Flow
1. **Camera** (camera.py) - RGB-D data acquisition via RealSense cameras
2. **Detection** (det_online.py) - Online object detection and grasp pose estimation
3. **Reference** (referring online) - Semantic object localization via natural language
4. **Robot Control** (robot_xarm.py/robot_piper.py) - Robotic arm control with coordinate transformations
5. **Demo Integration** (demo.py) - Main orchestrator coordinating all modules
6. **Visualization** (vis_cls.py) - Real-time display and logging
7. **Voice Interface** (voice_input.py) - Voice command processing

### Key Components

- **src/demo.py** - Main control loop and system integration
- **src/camera.py** - RealSense camera interface with RGB-D processing
- **src/det_online.py** - GraspAnythingOnline detection service
- **src/robot_xarm.py** - XArm robotic arm controller
- **src/robot_piper.py** - Piper arm controller (XArm-compatible interface)
- **src/piper_api.py** - Piper SDK wrapper providing XArmAPI compatibility
- **src/vis_cls.py** - Visualization and data logging
- **src/voice_input.py** - Voice command interface

### Coordinate System Transformations
The system implements a complete coordinate transformation chain:
- Camera coordinates → Flange coordinates → Gripper coordinates → Base coordinates
- Configuration files: `src/extrinsics_flan_to_hand_camera.yaml`

## Development Commands

### Hardware Setup
```bash
# CAN interface setup for Piper robots
cd src/
bash ./bringup.sh

# Manual CAN setup (if needed)
bash ./can_activate.sh can0 1000000 1-4.2.3:1.0  # for piper
bash ./can_activate.sh can1 500000 1-4.2.4:1.0   # for chassis
```

### Running the System
```bash
cd src/
export PATH="/usr/bin:$PATH" && python3 demo.py
```

### Testing Individual Components
```bash
# Test Piper API compatibility
python3 test_piper_api.py

# Test camera functionality
python3 -c "from camera import RealSenseCamera; print('Camera OK')"

# Verify Piper configuration
python3 verify_piper_config.py
```

## System Configuration

### Robot Platform Selection
- **XArm**: Use `robot_xarm.py` (line 17 in demo.py)
- **Piper**: Use `robot_piper.py` (line 18 in demo.py, currently commented)

### Critical Configuration Files
- **src/server_grasp.json** - Detection service backends
- **src/extrinsics_flan_to_hand_camera.yaml** - Camera-to-robot calibration
- **src/mobile_manipulator2_description.urdf** - Robot description

## Development Guidelines

### Code Organization
- **Main modules**: Place core functionality in `src/`
- **Test scripts**: Use `_cc_` prefix for temporary test scripts
- **Dependencies**: Piper SDK embedded in `src/piper_sdk/`
- **XArm SDK**: External dependency in `src/xArm-Python-SDK/`

### Robot Interface Compatibility
Both robot controllers (XArm and Piper) implement identical interfaces:
- `connect()` - Initialize robot connection
- `set_position()` - Cartesian position control
- `get_position()` - Current pose feedback
- `set_gripper_position()` - Gripper control
- Coordinate transformation methods

### Hardware Dependencies
- **RealSense Camera**: pyrealsense2 library
- **CAN Interface**: Required for Piper robots
- **Network**: Required for XArm robots (TCP/IP)

### Key Libraries
- **Computer Vision**: OpenCV, pyrealsense2
- **Robotics**: xarm-python-sdk, custom piper_sdk
- **AI/ML**: mmengine for configuration management
- **3D Processing**: scipy.spatial.transform for rotations

## Environment Setup

### Python Path Configuration
Always use: `export PATH="/usr/bin:$PATH" && python3`

### System Requirements
- ROS environment (though not using catkin/colcon)
- CAN utilities for hardware communication
- Intel RealSense SDK
- CUDA (likely for detection models)

## Safety and Hardware Considerations

- **CAN Setup**: Critical for Piper robot communication
- **Camera Calibration**: Hand-eye calibration required via YAML config
- **Coordinate Frames**: Multiple transformations between camera, flange, gripper, and base
- **Error Handling**: Robot error clearing and state management

## File Naming Conventions

- **Core modules**: Direct names (camera.py, robot_xarm.py)
- **Test/verification**: Descriptive names with purpose (verify_piper_config.py)
- **Temporary scripts**: Use `_cc_` prefix
- **Configuration**: Use descriptive YAML/JSON names

## Common Issues

- **Permission Issues**: Use `sudo` password: `didi`
- **CAN Interface**: Requires proper hardware setup via bringup.sh
- **ROS Environment**: Clean existing processes before running new ones
- **Camera Dependencies**: RealSense drivers must be properly installed

## API Compatibility Layer

The system provides seamless switching between robot platforms through:
- **PiperAPI**: XArmAPI-compatible wrapper for Piper robots
- **PiperRobot**: XArmRobot-compatible interface
- **Identical method signatures**: Enables drop-in replacement