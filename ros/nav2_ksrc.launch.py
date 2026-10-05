"""KSRC 로버 + Nav2 최소 구성 런치.

측위는 브리지가 map->odom 을 **항등으로** 발행하므로 AMCL 을 띄우지 않는다.
그래서 "Nav2 가 우리 로버에 맞게 도는가" 만 남고 측위 문제가 섞이지 않는다.
실제 측위(D435 지형 ICP)가 되면 그 항등 변환만 갈아끼우면 된다.

인지가 없으므로 costmap 은 static + inflation 뿐이다. 정적 맵은 경사·단차의
통행 불가 영역을 표시하고, 실제 로버 크기는 0.19m 풋프린트로 반영한다.

사용 (브리지 + Nav2 + RViz 를 함께 실행):
  ros2 launch nav2_ksrc.launch.py                       # 규사
  ros2 launch nav2_ksrc.launch.py terrain:=rock
  ros2 launch nav2_ksrc.launch.py rviz:=false
  ros2 launch nav2_ksrc.launch.py bridge:=false          # 브리지를 따로 실행할 때
  ros2 launch nav2_ksrc.launch.py cmd_filter_tau:=0      # cmd 필터 비교군
  ros2 launch nav2_ksrc.launch.py rl:=true                # s2 정책 잔차 적용
  ros2 launch nav2_ksrc.launch.py mode:=teleop           # 대회맵 + 브리지 + RViz

시뮬 전용 ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1 을 모든 자식 프로세스에
설정해 다른 ROS 실행의 /odom, /map_server 와 섞이지 않게 한다.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def setup(ctx, *args, **kw):
    terrain = LaunchConfiguration("terrain").perform(ctx)
    mode = LaunchConfiguration("mode").perform(ctx)
    use_rviz = LaunchConfiguration("rviz").perform(ctx).lower() in ("1", "true", "yes")
    use_bridge = LaunchConfiguration("bridge").perform(ctx).lower() in ("1", "true", "yes")
    use_view = LaunchConfiguration("view").perform(ctx).lower() in ("1", "true", "yes")
    use_rl = LaunchConfiguration("rl").perform(ctx).lower() in ("1", "true", "yes")
    use_teleop_guard = LaunchConfiguration("teleop_guard").perform(ctx).lower() in ("1", "true", "yes")
    cmd_filter_tau = LaunchConfiguration("cmd_filter_tau").perform(ctx)
    cmd_timeout = LaunchConfiguration("cmd_timeout").perform(ctx)
    drive_align_gate_deg = LaunchConfiguration("drive_align_gate_deg").perform(ctx)
    rl_model = LaunchConfiguration("rl_model").perform(ctx)
    rl_vecnorm = LaunchConfiguration("rl_vecnorm").perform(ctx)
    rl_scale = LaunchConfiguration("rl_scale").perform(ctx)
    mech_fold = LaunchConfiguration("mech_fold").perform(ctx).lower() in ("1", "true", "yes")
    steer_limit = LaunchConfiguration("steer_limit_deg").perform(ctx)
    domain_id = LaunchConfiguration("domain_id").perform(ctx)
    if terrain not in ("sand", "rock"):
        raise RuntimeError(f"알 수 없는 지형: {terrain}. sand 또는 rock 을 사용하세요.")
    if mode not in ("nav2", "teleop"):
        raise RuntimeError("mode 는 nav2 또는 teleop 이어야 합니다")
    if mode == "teleop" and use_rl:
        raise RuntimeError("teleop 모드에는 /plan 이 없으므로 rl:=true 를 쓸 수 없습니다")
    try:
        tau = float(cmd_filter_tau)
    except ValueError as exc:
        raise RuntimeError("cmd_filter_tau 는 0 이상의 초 단위 숫자여야 합니다") from exc
    if not (0.0 <= tau < float("inf")):
        raise RuntimeError("cmd_filter_tau 는 0 이상의 유한한 숫자여야 합니다")
    try:
        timeout = float(cmd_timeout)
    except ValueError as exc:
        raise RuntimeError("cmd_timeout 은 0보다 큰 초 단위 숫자여야 합니다") from exc
    if not (0.0 < timeout < float("inf")):
        raise RuntimeError("cmd_timeout 은 0보다 큰 유한한 숫자여야 합니다")
    try:
        gate_deg = float(drive_align_gate_deg)
    except ValueError as exc:
        raise RuntimeError("drive_align_gate_deg 는 0 이상의 숫자여야 합니다") from exc
    if not (0.0 <= gate_deg < float("inf")):
        raise RuntimeError("drive_align_gate_deg 는 0 이상의 유한한 숫자여야 합니다")
    try:
        scale = float(rl_scale)
    except ValueError as exc:
        raise RuntimeError("rl_scale 은 0 이상 1 이하 숫자여야 합니다") from exc
    if not (0.0 <= scale <= 1.0):
        raise RuntimeError("rl_scale 은 0 이상 1 이하 숫자여야 합니다")
    params = os.path.join(HERE, "nav2_params.yaml")
    map_yaml = os.path.join(HERE, "maps", f"arena_{terrain}.yaml")
    if not os.path.exists(map_yaml):
        raise RuntimeError(
            f"맵이 없다: {map_yaml}\n"
            f"먼저 실행하세요: cd {HERE} && python3 make_maps.py")

    nodes = [
        SetEnvironmentVariable("ROS_DOMAIN_ID", domain_id),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
    ]
    if use_bridge:
        bridge_cmd = [sys.executable, "-u", os.path.join(HERE, "mujoco_bridge.py"),
                      "--terrain", terrain, "--cmd-filter-tau", cmd_filter_tau,
                      "--cmd-timeout", cmd_timeout,
                      "--drive-align-gate-deg", drive_align_gate_deg]
        if mode == "teleop" and use_teleop_guard:
            bridge_cmd.append("--teleop-guard")
        if mech_fold:
            bridge_cmd.append("--mech-fold")
        if steer_limit:
            try:
                lim = float(steer_limit)
            except ValueError as exc:
                raise RuntimeError("steer_limit_deg 는 숫자여야 합니다") from exc
            if not (90.0 <= lim <= 180.0):
                raise RuntimeError("steer_limit_deg 는 90~180 사이여야 합니다 "
                                   "(90 미만이면 낼 수 없는 방향이 생깁니다)")
            bridge_cmd += ["--steer-limit-deg", steer_limit]
        if use_rl:
            if not os.path.isfile(rl_model) or not os.path.isfile(rl_vecnorm):
                raise RuntimeError(f"RL 모델 또는 vecnorm 이 없음: {rl_model}, {rl_vecnorm}")
            bridge_cmd += ["--model", rl_model, "--vecnorm", rl_vecnorm,
                           "--rl-scale", rl_scale]
        if use_view:
            bridge_cmd.append("--view")
        nodes.append(ExecuteProcess(cmd=bridge_cmd, name="mujoco_bridge", output="screen"))
    nodes += [
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen",
             parameters=[params, {"use_sim_time": False, "yaml_filename": map_yaml}]),
    ]
    if mode == "nav2":
        nodes += [
            Node(package="nav2_planner", executable="planner_server", name="planner_server",
                 output="screen", parameters=[params, {"use_sim_time": False}]),
            Node(package="nav2_controller", executable="controller_server",
                 name="controller_server", output="screen",
                 parameters=[params, {"use_sim_time": False}],
                 # velocity_smoother 를 안 쓰므로 컨트롤러가 /cmd_vel 로 직접 낸다
                 remappings=[("cmd_vel", "cmd_vel")]),
            Node(package="nav2_behaviors", executable="behavior_server",
                 name="behavior_server", output="screen",
                 parameters=[params, {"use_sim_time": False}]),
            Node(package="nav2_bt_navigator", executable="bt_navigator",
                 name="bt_navigator", output="screen",
                 parameters=[params, {"use_sim_time": False}]),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_navigation", output="screen",
                 parameters=[{"use_sim_time": False, "autostart": True,
                              "node_names": ["map_server", "planner_server",
                                             "controller_server", "behavior_server",
                                             "bt_navigator"]}]),
        ]
    else:
        nodes.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                          name="lifecycle_manager_map", output="screen",
                          parameters=[{"use_sim_time": False, "autostart": True,
                                       "node_names": ["map_server"]}]))
    if use_rviz:
        cfg = os.path.join(HERE, "rviz", "teleop_ksrc.rviz" if mode == "teleop"
                           else "ksrc.rviz")
        nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2",
                          output="screen",
                          arguments=(["-d", cfg] if os.path.exists(cfg) else []),
                          parameters=[{"use_sim_time": False}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("terrain", default_value="sand",
                              description="sand | rock"),
        DeclareLaunchArgument("mode", default_value="nav2",
                              description="nav2 | teleop (map + bridge + RViz only)"),
        DeclareLaunchArgument("teleop_guard", default_value="true",
                              description="teleop 에서 큰 방향 전환 시 차체 감속"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("bridge", default_value="true"),
        DeclareLaunchArgument("view", default_value="true"),
        DeclareLaunchArgument("cmd_filter_tau", default_value="0.10",
                              description="브리지 cmd_vel 저역통과 시정수 [s], 0=끔"),
        DeclareLaunchArgument("cmd_timeout", default_value="0.5",
                              description="브리지 /cmd_vel 수신 만료 후 정지 [s]"),
        DeclareLaunchArgument("drive_align_gate_deg", default_value="10.0",
                              description="전 모드 공통 구동 허용 조향 오차 [deg], 0=끔"),
        DeclareLaunchArgument("rl", default_value="false",
                              description="브리지에서 3D student RL 잔차를 적용"),
        # 2026-10-04: s2 -> s13 @2.0M.  s2 는 조향 +-180도, 옛 보상, 옛 e_psi 기준에서
        # 학습해 현 환경과 맞지 않았다.  s13 @2.0M 은 holdout 100 에피소드 짝비교에서
        # 순수 IK 를 **유의하게 이긴 유일한 체크포인트**다 (잡음 ON 조건):
        #   횡오차 21.68 -> 19.80mm (-8.7%),  p90 46.58 -> 42.34mm (-9.1%)
        #   기수 7.38 -> 7.12도 (-3.5%),      완주율 95% 유지
        # 다른 시드(seed0=2000)·다른 경로세트(path-seed=7777)에서도 -10.0% / -8.6% 로
        # 재현된다.  대가: 슬립 +26%, 소요시간 +3.8%, cmd_sat 4.6배.
        # **잡음 OFF 조건에서는 중립적**이다 (평균 횡오차 +4.9% 비유의, p90 -7.6% 유의).
        DeclareLaunchArgument("rl_model", default_value=os.path.join(
            HERE, "..", "sim", "rl", "runs", "s13_resid",
            "ppo_1999900_steps.zip")),
        DeclareLaunchArgument("rl_vecnorm", default_value=os.path.join(
            HERE, "..", "sim", "rl", "runs", "s13_resid",
            "ppo_vecnormalize_1999900_steps.pkl")),
        DeclareLaunchArgument("rl_scale", default_value="1.0",
                              description="정책 잔차 배율 [0,1]"),
        DeclareLaunchArgument("mech_fold", default_value="false",
                              description="접기를 '모터가 차체 중심에 가까운 쪽' 으로"),
        DeclareLaunchArgument("steer_limit_deg", default_value="",
                              description="조향 가동범위 ±[deg], 빈 값이면 config 기본(±100)"),
        DeclareLaunchArgument("domain_id", default_value="77"),
        OpaqueFunction(function=setup),
    ])
