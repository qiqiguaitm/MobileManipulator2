# 碰撞模型工具

这个目录包含了用于生成和管理mobile_manipulator2机器人碰撞模型的工具。

## 工具列表

### 1. generate_collision_models.py
生成mobile_manipulator2的简化碰撞模型。支持凸包、包围盒、圆柱体等不同的简化策略。

### 2. generate_piper_collision_models.py  
专门用于生成Piper机械臂各个关节的简化碰撞模型。

### 3. simplify_collision_models.py
批量简化STL模型文件，支持多种简化算法。

### 4. update_performance_urdf.py
更新性能版URDF文件，将碰撞模型路径指向简化版本。

### 5. update_piper_collision.py
更新Piper机械臂的碰撞模型配置。

### 6. update_urdf_local_models.py
确保URDF文件使用本地碰撞模型而不是外部引用。

### 7. verify_collision_models.sh
验证碰撞模型是否正确生成和配置。

### 8. test_collision_models.sh
测试碰撞模型在MoveIt中的加载和性能。

## 使用说明

1. 生成碰撞模型：
```bash
python3 generate_collision_models.py
```

2. 验证模型：
```bash
./verify_collision_models.sh
```

3. 测试性能：
```bash
./test_collision_models.sh
```

## 注意事项

- 生成碰撞模型时已经移除了padding，所有安全间距由MoveIt的collision_padding.yaml配置
- 确保在修改碰撞模型后重新编译工作空间