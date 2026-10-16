"""RViz 에서 목표를 찍어 Nav2 로 몰며 IK 와 Isaac student(RL) 를 비교하는 런치 (2026-10-15).

MuJoCo rev5 (원기둥 바퀴, 펌웨어 IK·게이트·바퀴 PI, 실기식 센서·지연·잡음) + Isaac 정책 브리지(isaac_bridge.py)
+ Nav2 (MPPI Omni) + RViz (rev5 로버 모델).  측위는 참값.  옛 nav2_ksrc.launch.py(SB3 정책)는 그대로 둔다.

  ros2 launch ~/ksrc/ros/nav2_isaac.launch.py policy:=<run>/model_1000.pt mu:=const:0.3
  # 주행 중 비교 전환:  ros2 service call /set_rl std_srvs/srv/SetBool "{data: false}"   (IK)
  #                    ros2 service call /set_rl std_srvs/srv/SetBool "{data: true}"    (RL)
  # 같은 출발점에서 다시:  ros2 service call /reset_sim std_srvs/srv/Trigger
  (같은 ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1 셸에서)
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
DESC = os.path.join(HERE, "rover_description")


def setup(ctx):
    g = lambda k: LaunchConfiguration(k).perform(ctx)
    terrain, policy, mu, soil = g("terrain"), g("policy"), g("mu"), g("soil")
    on = lambda k: g(k).lower() in ("1", "true", "yes")
    if policy and not os.path.isfile(policy):
        raise RuntimeError(f"정책 파일 없음: {policy}")
    params = os.path.join(HERE, "nav2_params.yaml")
    map_yaml = os.path.join(HERE, "maps", f"arena_{terrain}.yaml")
    urdf = open(os.path.join(DESC, "rover_rev5.urdf")).read().replace('filename="meshes/', f'filename="file://{DESC}/meshes/')
    bridge = [sys.executable, "-u", os.path.join(HERE, "isaac_bridge.py"), "--terrain", terrain, "--mu", mu,
              "--soil", soil]
    if policy:
        bridge += ["--policy", policy]
    bridge += ["--gate", g("gate"), "--vo", g("vo")]
    nav_model, smoother = g("nav_model"), on("smoother")
    if nav_model not in ("rpp", "diff", "omni"):
        raise RuntimeError("nav_model 은 rpp, diff 또는 omni")
    # 10-15: Omni 는 옆걸음 명령을 샘플링 잡음과 함께 내서, ±90° 접기 경계에서 조향이 매 주기 뒤집혀
    # (바퀴 방향도 같이 뒤집힘) 제자리에 선다 (MuJoCo 재현: 게이트 old/new/off 모두).  diff = 몸을 돌려 전진.
    ctrl_over = {"use_sim_time": False}
    if nav_model == "diff":
        # TwirlingCritic(제자리 회전 벌점 10)은 Omni 용: DiffDrive 는 몸을 먼저 돌려야 해서 이게 회전을 막아
        # 명령이 거의 0 이 됐다 (10-15 시험: 20 s 동안 "Failed to make progress").  끄고 경로 방향 비중을 올린다.
        ctrl_over.update({"FollowPath.motion_model": "DiffDrive", "FollowPath.vx_max": 0.25,
                          "FollowPath.vx_min": -0.15, "FollowPath.wz_max": 0.8,
                          "FollowPath.TwirlingCritic.enabled": False,
                          "FollowPath.PathAngleCritic.cost_weight": 6.0})
    if nav_model == "rpp":
        # 10-15: 제안서 구조 (RPP: v, κ -> ω = vκ, vy 없음).  결정론이라 MPPI 처럼 명령이 매 주기 흔들리지 않는다.
        # RotationShim: 경로 방향과 45° 넘게 어긋나면 먼저 제자리 회전, 그다음 RPP.
        P = "FollowPath."
        ctrl_over.update({
            P + "plugin": "nav2_rotation_shim_controller::RotationShimController",
            P + "primary_controller": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
            P + "angular_dist_threshold": 0.785, P + "forward_sampling_distance": 0.3,
            P + "rotate_to_heading_angular_vel": 0.5, P + "max_angular_accel": 2.0, P + "simulate_ahead_time": 1.0,
            P + "desired_linear_vel": 0.15, P + "lookahead_dist": 0.35, P + "min_lookahead_dist": 0.25,
            P + "max_lookahead_dist": 0.6, P + "lookahead_time": 1.5, P + "use_velocity_scaled_lookahead_dist": True,
            P + "transform_tolerance": 0.2, P + "min_approach_linear_velocity": 0.05,
            P + "approach_velocity_scaling_dist": 0.4, P + "use_collision_detection": False,   # 전역 계획기가 이미 회피; 정적 맵의 경사면 가장자리를 충돌로 봐 멈췄음
            P + "max_allowed_time_to_collision_up_to_carrot": 1.0, P + "use_regulated_linear_velocity_scaling": True,
            P + "use_cost_regulated_linear_velocity_scaling": False, P + "regulated_linear_scaling_min_radius": 0.6,
            P + "regulated_linear_scaling_min_speed": 0.08, P + "use_rotate_to_heading": False,
            P + "allow_reversing": False, P + "max_robot_pose_search_dist": 2.0})
    ctrl_remap = [("cmd_vel", "cmd_vel_nav")] if smoother else [("cmd_vel", "cmd_vel")]
    if on("start_ik"):
        bridge.append("--start-ik")
    if on("view"):
        bridge.append("--view")
    nodes = [
        SetEnvironmentVariable("ROS_DOMAIN_ID", g("domain_id")),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        ExecuteProcess(cmd=bridge, name="isaac_bridge", output="screen"),
        Node(package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher",
             output="screen", parameters=[{"robot_description": urdf, "use_sim_time": False}]),
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[params, {"use_sim_time": False, "yaml_filename": map_yaml}]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[params, {"use_sim_time": False}]),
        Node(package="nav2_controller", executable="controller_server", name="controller_server", output="screen",
             parameters=[params, ctrl_over], remappings=ctrl_remap),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server", output="screen",
             parameters=[params, {"use_sim_time": False}]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator", output="screen",
             parameters=[params, {"use_sim_time": False}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
             output="screen", parameters=[{"use_sim_time": False, "autostart": True,
                                           "node_names": ["map_server", "planner_server", "controller_server",
                                                          "behavior_server", "bt_navigator"]
                                           + (["velocity_smoother"] if smoother else [])}]),
    ]
    if smoother:   # 학습과 같은 가속 제한 (rover_env_cfg CMD_ACC_LIN 0.5 m/s², CMD_ACC_ANG 2 rad/s²)
        nodes.append(Node(package="nav2_velocity_smoother", executable="velocity_smoother", name="velocity_smoother",
                          output="screen",
                          parameters=[{"use_sim_time": False, "smoothing_frequency": 50.0, "scale_velocities": False,
                                       "feedback": "OPEN_LOOP", "max_velocity": [0.25, 0.20, 0.8],
                                       "min_velocity": [-0.25, -0.20, -0.8], "max_accel": [0.5, 0.5, 2.0],
                                       "max_decel": [-0.5, -0.5, -2.0], "deadband_velocity": [0.0, 0.0, 0.0],
                                       "velocity_timeout": 1.0}],
                          remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")]))
    if on("rviz"):
        nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
                          arguments=["-d", os.path.join(HERE, "rviz", "ksrc_isaac.rviz")],
                          parameters=[{"use_sim_time": False}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("terrain", default_value="sand", description="sand | rock (대회 STL 높이맵)"),
        DeclareLaunchArgument("policy", default_value="", description="Isaac student 체크포인트 (params/ 같이). 빈 값 = IK 만"),
        DeclareLaunchArgument("mu", default_value="const:0.5", description="const:μ (4바퀴 같음) | uniform:lo,hi | patch"),
        DeclareLaunchArgument("soil", default_value="rigid", description="rigid | sand (MuJoCo 지반역학)"),
        DeclareLaunchArgument("start_ik", default_value="false", description="정책을 불러오되 IK 로 시작"),
        DeclareLaunchArgument("vo", default_value="dr",
                              description="student v3 VO 관측: dr = 학습 VO 모델 (지연·잡음·끊김) | off = VO 없음 (Pi 기본)"),
        DeclareLaunchArgument("gate", default_value="new",
                              description="구동 정렬 게이트: new (현 펌웨어·Isaac 학습, 기본) | old (10-13 이전 펌웨어) | off"),
        DeclareLaunchArgument("view", default_value="true", description="MuJoCo 창도 띄움"),
        DeclareLaunchArgument("nav_model", default_value="rpp",
                              description="rpp (RotationShim + Regulated Pure Pursuit, 제안서 구조, 기본) | diff (MPPI DiffDrive) | omni (MPPI 옆걸음: 접기 경계 문제)"),
        DeclareLaunchArgument("smoother", default_value="true", description="Nav2 velocity_smoother (가속 제한)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("domain_id", default_value="77"),
        OpaqueFunction(function=setup),
    ])
