#!/usr/bin/env bash
# Run from anywhere: live visual identity plus LiDAR geometry, never a saved target pose.
set -eo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=--execute
selected_mode=
selected_detection=
forward=()
for argument in "$@"; do
  case "$argument" in
    --help|-h)
      cat <<'HELP'
用法：go_to_object.sh（也可用 go_to_board.sh） [--check] [--standoff 米数] [--target 目标描述] [--aruco | --no-aruco]

默认：--no-aruco --detector deepseek --geometry-source lidar，相机识别物体正面，LiDAR 拟合平面，沿法向停靠。
--target：目标描述，Qwen/DeepSeek/GPT 支持中文；默认窗边白色方柱右侧、地面上白色空气净化器朝向机器人的完整正面。
--detector deepseek|qwen|gpt|dinox：默认 DeepSeek，直接联网、不使用 Mac 代理。
--startup-observations 2|3：默认在最多 10 帧中收集 2 帧有效观测，保留位置及法向一致性校验。
--qwen-model：默认 qwen3-vl-plus，使用独立 DashScope API key，直连。
--deepseek-model：默认 deepseek-flash，需要单独配置 DeepSeek API key。
--gpt-model：可覆盖配置中的模型，缺省 gpt-4.1。
--detector-url：仅 DINO-X 使用，也可设置 OBJECT_DETECTOR_URL。
--aruco：使用当前 7×5 标定板编号检测，不调用模型 API，缺省使用双目深度；同样沿板面法向停靠。
--no-aruco：默认，物体检测＋LiDAR，不依赖棋盘格或标记。
--geometry-source lidar|depth：显式选择雷达或双目测距；不会在失败后静默切换。
--check：观察、规划并验证异步反馈，不启用底盘。
--standoff：底盘参考点到目标正面平面的水平法向距离，0.5～2.0 米，默认 0.934 米（按车头前伸 0.434 米计算，车头间隙约 0.5 米）。

运行前定位与 Nav2 必须就绪，请留在现场并保持急停可用。
Ctrl+C 中止任务。结果位于项目 log/board_approach/<时间>/result.json。
全程持续异步检测并更新最终停靠目标，不设置中间对准点；终点优先复用新鲜测量验收，缺失时补测一帧。
安全故障立即停车；不保证障碍物或控制器不会要求停车。
每次使用实时相机和所选测距传感器，不读取预存物体坐标，不需人工框选。
LiDAR 模式保存检测框、雷达投影、平面内点/剔除点及法向图片；测量不足立即拒绝，不猜测深度。
HELP
      exit 0
      ;;
    --check|--execute)
      if [[ -n "$selected_mode" && "$selected_mode" != "$argument" ]]; then
        echo '--check and --execute cannot be combined' >&2
        exit 2
      fi
      selected_mode="$argument"
      mode="$argument"
      ;;
    --aruco|--no-aruco)
      if [[ -n "$selected_detection" && "$selected_detection" != "$argument" ]]; then
        echo '--aruco and --no-aruco cannot be combined' >&2
        exit 2
      fi
      selected_detection="$argument"
      forward+=("$argument")
      ;;
    *) forward+=("$argument") ;;
  esac
done
exec bash "$script_dir/approach_board.sh" "$mode" "${forward[@]}"
