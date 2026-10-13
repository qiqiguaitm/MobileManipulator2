# Autonomous exploration

- `auto_explore/`: exploration orchestration, occupancy mapping from SC-PGO and FAST-LIO poses, motion safety, and final map export.
- `m-explore-ros2/`: vendored upstream exploration code (`explore_lite`) and its status messages (`explore_lite_msgs`).

SLAM algorithms remain in `src/fast_lio/` and `src/sc_pgo/`; shared SLAM integration remains in `src/slam/`.

Build from the repository root:

```bash
colcon build --symlink-install --packages-select explore_lite_msgs explore_lite auto_explore
source install/local_setup.bash
```

The entry point remains `bash scripts/auto_explore.sh`; see [auto_explore documentation](auto_explore/README.md).
