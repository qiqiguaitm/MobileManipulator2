"""Standalone mapping stack. Must not coexist with HDL saved-map navigation."""
from pathlib import Path
from auto_explore.geometry import footprint as chassis_footprint
from auto_explore.geometry import PLANNING_PADDING, GOAL_XY_TOLERANCE
import yaml
import json
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory as share


def setup(context):
    output=Path(LaunchConfiguration('output').perform(context));output.mkdir(parents=True,exist_ok=True)
    execute=LaunchConfiguration('execute').perform(context)=='true'
    slam=Path(share('slam'));ours=Path(share('auto_explore'))
    cfg=yaml.safe_load((slam/'config/nav2_minimal_params.yaml').read_text())
    cfg.pop('map_server',None);cfg.pop('amcl',None)
    controller=cfg['controller_server']['ros__parameters'];controller['odom_topic']='/odom'
    # Use the installed Nav2 controller for this independent mode; existing TEB
    # crashes on teardown in the hardware-free lifecycle smoke test.
    defaults=yaml.safe_load((Path(share('nav2_bringup'))/'params/nav2_params.yaml').read_text())
    controller['FollowPath']=defaults['controller_server']['ros__parameters']['FollowPath']
    controller['FollowPath'].update(max_vel_x=.10,max_vel_y=0.,max_speed_xy=.10,max_vel_theta=.18,
        min_vel_x=0.,acc_lim_x=.2,decel_lim_x=-.2,acc_lim_theta=.4,decel_lim_theta=-.4,
        trans_stopped_velocity=.02,xy_goal_tolerance=GOAL_XY_TOLERANCE,debug_trajectory_details=False,
        publish_evaluation=True,short_circuit_trajectory_evaluation=False,
        vx_samples=10,vy_samples=1,vtheta_samples=15,
        critics=['RotateToGoal','Oscillation','ObstacleFootprint','GoalAlign','PathAlign','PathDist','GoalDist'])
    # Collision rejection is independent of score weight. Match Nav2's
    # normal obstacle-cost scale so inflation does not make standing still win.
    controller['FollowPath']['ObstacleFootprint.scale']=.02
    # A rearward NavFn path can make DWB's forward-point alignment cost prefer
    # standing still to starting a turn. Use Nav2's collision-checked heading
    # shim before path tracking instead of weakening DWB's collision checks.
    # The primary controller shares the FollowPath parameter namespace.
    controller['FollowPath'].update(
        plugin='nav2_rotation_shim_controller::RotationShimController',
        primary_controller='dwb_core::DWBLocalPlanner',
        angular_dist_threshold=.35,angular_disengage_threshold=.35,
        forward_sampling_distance=.5,rotate_to_heading_angular_vel=.18,
        # FAST-LIO /odom supplies pose but leaves twist zero. Ramp from the
        # previous command, otherwise each cycle is clamped back to 0.05 rad/s.
        # Heading feedback still uses measured pose/TF, not integrated commands.
        max_angular_accel=.4,simulate_ahead_time=2.,closed_loop=False,
        rotate_to_goal_heading=False)
    # Turning toward the path is real progress; still time out a stationary
    # controller after 20 seconds. Do not relax the translation requirement.
    controller['progress_checker'].update(
        plugin='nav2_controller::PoseProgressChecker',
        required_movement_angle=.15,required_movement_radius=.10,
        movement_time_allowance=20.)
    controller['goal_checker'].update(xy_goal_tolerance=GOAL_XY_TOLERANCE,yaw_goal_tolerance=.25)
    for name in ('local','global'):
        c=cfg[name+'_costmap'][name+'_costmap']['ros__parameters']
        c['track_unknown_space']=True;c['resolution']=.05
        c['footprint']=json.dumps(chassis_footprint())
        c['footprint_padding']=PLANNING_PADDING
        c['obstacle_layer']['track_unknown_space']=True;c['obstacle_layer']['scan']['topic']='/explore/scan'
        c['obstacle_layer']['scan']['inf_is_valid']=False
        c['inflation_layer']['inflation_radius']=.85
    # A full raw grid is published for the read-only overlay; planning is unchanged.
    cfg['local_costmap']['local_costmap']['ros__parameters']['publish_frequency']=4.
    cfg['global_costmap']['global_costmap']['ros__parameters']['static_layer'].update(
        track_unknown_space=True,subscribe_to_updates=True)
    # Use the same accumulated map for local collision checks. Scan-only local
    # maps leave gaps between LiDAR rays unknown, even where the global mapper
    # has already observed free space. Live scan obstacles remain overlaid.
    local=cfg['local_costmap']['local_costmap']['ros__parameters']
    local['plugins']=['static_layer','obstacle_layer','inflation_layer']
    local['static_layer']=dict(cfg['global_costmap']['global_costmap']['ros__parameters']['static_layer'])
    cfg['bt_navigator']['ros__parameters'].update(odom_topic='/odom',default_nav_to_pose_bt_xml=str(ours/'config/explore.xml'),default_nav_through_poses_bt_xml=str(ours/'config/explore.xml'))
    # No blind backup/spin recovery or clear-costmap action; coordinator chooses another frontier.
    config=output/'nav2.yaml';config.write_text(yaml.safe_dump(cfg))
    bounds=json.loads(LaunchConfiguration('bounds').perform(context))
    upstream={'/**':{'ros__parameters':dict(robot_base_frame='base_link',return_to_init=True,
        costmap_topic='/global_costmap/costmap',costmap_updates_topic='/global_costmap/costmap_updates',
        visualize=True,planner_frequency=.5,progress_timeout=30.,potential_scale=3.,orientation_scale=0.,
        gain_scale=1.,transform_tolerance=.3,min_frontier_size=.5)}}
    (output/'explore_lite.yaml').write_text(yaml.safe_dump(upstream))
    urdf=(Path(share('mobile_manipulator2_description'))/'urdf/mobile_manipulator2_description.urdf').read_text()
    def n(pkg,exe,name=None,params=None,remaps=None):
        return Node(package=pkg,executable=exe,name=name,parameters=params or [],remappings=remaps or [],output='screen')
    def tf(name,parent,child,xyz,quat=(0,0,0,1)):
        return Node(package='tf2_ros',executable='static_transform_publisher',name=name,
          arguments=['--x',str(xyz[0]),'--y',str(xyz[1]),'--z',str(xyz[2]),'--qx',str(quat[0]),'--qy',str(quat[1]),'--qz',str(quat[2]),'--qw',str(quat[3]),'--frame-id',parent,'--child-frame-id',child])
    nodes=[IncludeLaunchDescription(PythonLaunchDescriptionSource(str(slam/'launch/scpgo_mapping_launch.py')),
            launch_arguments={'enable_rviz':'false','save_dir':str(output/'sc_pgo')+'/'}.items()),
        n('slam','odom_frame_converter.py','odom_frame_converter',[{'input':'/fastlio/odom','output':'/odom'}]),
        n('robot_state_publisher','robot_state_publisher','robot_state_publisher',[{'robot_description':urdf}], [('/joint_states','/joint_states_static')]),
        n('slam','joint_state_static.py','joint_state_static',[{'topic':'/joint_states_static','rate':20.}]),
        tf('explore_body_base','body','base_link',(0,0,0)),
        tf('explore_lidar','lidar_link','rslidar',(0,0,.0635),(-.00218012051,.000590033588,0,.999997449)),
        n('auto_explore','cloud_filter','explore_cloud_filter'),
        n('pointcloud_to_laserscan','pointcloud_to_laserscan_node','explore_scan',[
            {'target_frame':'base_link','transform_tolerance':.2,'min_height':.04,'max_height':1.2,
             'angle_min':-3.14159,'angle_max':3.14159,'angle_increment':.00872,'scan_time':.1,
             'range_min':.2,'range_max':8.,'use_inf':True}], [('cloud_in','/explore/obstacle_cloud'),('scan','/explore/scan')]),
        n('auto_explore','exploration_mapper','exploration_mapper',[{'output':str(output/'occupancy'),'bounds':bounds,'resolution':.05}]),
        n('auto_explore','map_odom_tf','exploration_map_odom_tf'),
        n('nav2_planner','planner_server','planner_server',[str(config)]),
        n('nav2_controller','controller_server','controller_server',[str(config)],[('/cmd_vel','/explore/nav_cmd_vel')]),
        n('nav2_bt_navigator','bt_navigator','bt_navigator',[str(config)]),
        n('nav2_lifecycle_manager','lifecycle_manager','explore_lifecycle',[{'autostart':True,'node_names':['planner_server','controller_server','bt_navigator']}])]
    footprint=json.loads(cfg['local_costmap']['local_costmap']['ros__parameters']['footprint'])
    padded=[float(v+(PLANNING_PADDING if v>=0 else -PLANNING_PADDING)) for xy in footprint for v in xy]
    nodes.append(n('auto_explore','rotation_debug','explore_rotation_debug',
                   [{'output':str(output/'debug'),'footprint':padded}]))
    if execute:
        nodes.append(n('tracer_base','tracer_base_node','tracer',[
            {'port_name':'can2','publish_tf':False,'odom_topic_name':'/explore/wheel_odom',
             'odom_frame':'wheel_odom','command_timeout':.4}],[('/cmd_vel','/explore/safe_cmd_vel')]))
    return nodes


def generate_launch_description():
    return LaunchDescription([SetLaunchConfiguration('sigterm_timeout','35'),DeclareLaunchArgument('output'),DeclareLaunchArgument('execute',default_value='false'),DeclareLaunchArgument('bounds',default_value='[-25.0,-25.0,25.0,25.0]'),OpaqueFunction(function=setup)])
