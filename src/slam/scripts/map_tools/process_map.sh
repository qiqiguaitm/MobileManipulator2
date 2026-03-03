#!/bin/bash
# ============================================================
# SC-PGO 地图后处理一键脚本
# ============================================================
# 用法: ./process_map.sh [map_dir]
#
# 处理流程:
#   1. 合并Scans/*.pcd → GlobalMap.pcd
#   2. 自适应地面过滤 → GlobalMap_filtered.pcd
#   3. 降采样用于定位 → GlobalMap_hdl.pcd
#   4. 生成2D栅格地图 → map.pgm + map.yaml
#   5. 2D地图优化 → map.pgm (去噪)
# ============================================================

set -e

# 默认地图目录
MAP_DIR="${1:-/home/didi/workspace/MobileManipulator2/maps/sc_pgo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  SC-PGO 地图后处理"
echo "============================================================"
echo "地图目录: $MAP_DIR"
echo "脚本目录: $SCRIPT_DIR"
echo ""

# 检查地图目录
if [ ! -d "$MAP_DIR/Scans" ]; then
    echo "[ERROR] Scans目录不存在: $MAP_DIR/Scans"
    exit 1
fi

if [ ! -f "$MAP_DIR/optimized_poses.txt" ]; then
    echo "[ERROR] 位姿文件不存在: $MAP_DIR/optimized_poses.txt"
    exit 1
fi

# Step 1: 合并扫描帧
echo ""
echo "[Step 1/4] 合并扫描帧到GlobalMap.pcd..."
python3 "$SCRIPT_DIR/merge_scans.py" "$MAP_DIR" GlobalMap 0.02

# Step 2: 降采样用于HDL定位
echo ""
echo "[Step 2/4] 降采样用于定位..."
python3 "$SCRIPT_DIR/downsample_pcd.py" \
    "$MAP_DIR/GlobalMap.pcd" \
    "$MAP_DIR/GlobalMap_hdl.pcd" \
    0.1

# Step 3: 生成2D栅格地图
echo ""
echo "[Step 3/4] 生成2D栅格地图..."
python3 "$SCRIPT_DIR/pcd_to_2dmap.py" \
    "$MAP_DIR/GlobalMap.pcd" \
    "$MAP_DIR" \
    0.05

# Step 4: 2D地图优化
echo ""
echo "[Step 4/4] 2D地图优化..."
python3 "$SCRIPT_DIR/improve_2dmap.py" "$MAP_DIR"

echo ""
echo "============================================================"
echo "  地图处理完成!"
echo "============================================================"
echo ""
echo "输出文件:"
echo "  3D地图 (导航):  $MAP_DIR/GlobalMap.pcd"
echo "  3D地图 (定位):  $MAP_DIR/GlobalMap_hdl.pcd"
echo "  2D地图:         $MAP_DIR/map.pgm"
echo "  2D配置:         $MAP_DIR/map.yaml"
echo "  对比图:         $MAP_DIR/map_comparison.png"
echo ""
