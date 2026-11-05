"""XFeas ROS 2 node (Humble): runs the XFeas decision layer on a real robot.

    ros2 launch xfeas_ros g1_sonic.launch.py        # or spot_relic.launch.py

Subscribes (topic names in config/<robot>.yaml):
    joint_states  sensor_msgs/JointState     robot joint positions / velocities / efforts
    odom          nav_msgs/Odometry          base pose (world = odometry frame, z up) + body-frame twist
    rgb, depth    sensor_msgs/Image          head camera; depth ALIGNED to the colour image (16UC1 mm or 32FC1 m)
    camera_info   sensor_msgs/CameraInfo     intrinsics of the colour image
    goal          geometry_msgs/PoseStamped  goal position in the odometry frame (only x, y are used)
    enable        std_msgs/Bool              dead-man switch: commands are sent only while True
    TF            world <- camera optical frame, world <- body-top / foot frames
Publishes:
    command       xfeas_msgs/ControllerCommand   the controller's own command (twist + posture / mode), for the bridge
    cmd_vel       geometry_msgs/Twist            the same twist, for controllers that only take a twist
    status        std_msgs/String                one JSON line per decision (class, skill, score, top 3)

XFeas never drives the robot directly: a robot-side bridge turns ControllerCommand into the controller's input
(the GEAR-SONIC deploy stack on G1, the ReLIC policy on Spot). See ros2/README.md.
"""

import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from xfeas_msgs.msg import ControllerCommand

from xfeas_ros import core as XC

NAN = float("nan")


def image_to_numpy(msg):
    if msg.encoding in ("rgb8", "bgr8"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
        return a[..., ::-1] if msg.encoding == "bgr8" else a
    if msg.encoding in ("16UC1", "mono16"):
        a = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return a.astype(np.float32) / 1000.0  # millimetres -> metres
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def quat_xyzw(q):
    return (q.x, q.y, q.z, q.w)


class XFeasNode(Node):
    def __init__(self):
        super().__init__("xfeas")
        self.declare_parameter("config", "")
        path = self.get_parameter("config").get_parameter_value().string_value
        self.cfg = cfg = XC.load_config(path)
        self.topics = cfg["topics"]
        self.frames = cfg["frames"]
        self.core = XC.XFeasCore(cfg)
        self.core.start_model()
        self.joint_map = cfg.get("joint_map") or {}  # robot joint name -> model joint name
        self.idx = {n: i for i, n in enumerate(cfg["joint_names"])}
        self.q0 = np.asarray(cfg["default_pos"], float)
        self.resampler, self.enabled, self.t_sensor = None, False, {}
        self.max_vx = float(cfg.get("max_vx", 0.5))
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)

        T = self.topics
        self.create_subscription(JointState, T["joint_states"], self.on_joints, qos_profile_sensor_data)
        self.create_subscription(Odometry, T["odom"], self.on_odom, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, T["camera_info"], self.on_info, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, T["goal"], self.on_goal, 10)
        self.create_subscription(Bool, T["enable"], self.on_enable, 10)
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, T["rgb"], qos_profile=qos_profile_sensor_data),
             Subscriber(self, Image, T["depth"], qos_profile=qos_profile_sensor_data)], 10, 0.05)
        sync.registerCallback(self.on_rgbd)
        self.pub_cmd = self.create_publisher(ControllerCommand, T["command"], 10)
        self.pub_twist = self.create_publisher(Twist, T["cmd_vel"], 10)
        self.pub_status = self.create_publisher(String, T["status"], 10)
        self.create_timer(0.02, self.tick)  # 50 Hz: body state, decisions every 0.25 s, command out
        self.get_logger().info(f"XFeas ready: controller {cfg['controller_id']}, arm {cfg.get('arm')}, "
                               f"checkpoint {cfg['checkpoint']} (commands sent only while '{T['enable']}' is true)")

    # ------------------------------------------------------------------ helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def lookup(self, frame, stamp=None):
        """world <- frame as (R, t), or None"""
        try:
            tr = self.tf.lookup_transform(self.frames["world"], frame, stamp or Time())
        except Exception:
            return None
        q, p = tr.transform.rotation, tr.transform.translation
        return XC.rot_of_quat(quat_xyzw(q)), np.array([p.x, p.y, p.z])

    # ------------------------------------------------------------------ callbacks
    def on_joints(self, msg):
        q, dq, tau = self.q0.copy(), np.zeros_like(self.q0), np.zeros_like(self.q0)
        for k, n in enumerate(msg.name):
            i = self.idx.get(self.joint_map.get(n, n))
            if i is None:
                continue
            q[i] = msg.position[k] if k < len(msg.position) else q[i]
            dq[i] = msg.velocity[k] if k < len(msg.velocity) else 0.0
            tau[i] = msg.effort[k] if k < len(msg.effort) else 0.0
        self.t_sensor["joints"] = self.now()
        self.core.on_joints(self.now(), q, dq, tau)

    def on_odom(self, msg):
        p, tw = msg.pose.pose.position, msg.twist.twist  # twist in the child (body) frame (REP 105)
        self.t_sensor["odom"] = self.now()
        self.core.on_base(self.now(), (p.x, p.y, p.z), quat_xyzw(msg.pose.pose.orientation),
                          (tw.linear.x, tw.linear.y, tw.linear.z), (tw.angular.x, tw.angular.y, tw.angular.z))

    def on_info(self, msg):
        if self.resampler is None:
            self.resampler = XC.CameraResampler(np.asarray(msg.k).reshape(3, 3), msg.width, msg.height)
            self.get_logger().info(f"camera {msg.width}x{msg.height}: covers {100 * self.resampler.coverage:.0f} % of "
                                   f"the simulator camera image (the rest is blank)")

    def on_rgbd(self, rgb_msg, depth_msg):
        if self.resampler is None:
            return
        pose = self.lookup(self.frames["camera_optical"], Time.from_msg(rgb_msg.header.stamp))
        if pose is None:
            return
        rgb = self.resampler.rgb(image_to_numpy(rgb_msg))
        dep = self.resampler.depth(image_to_numpy(depth_msg))
        self.t_sensor["camera"] = self.now()
        self.core.on_frame(self.now(), rgb, dep, XC.camera_matrix_from_optical(*pose))

    def on_goal(self, msg):
        self.core.reset()
        self.core.set_goal((msg.pose.position.x, msg.pose.position.y))
        self.get_logger().info(f"new goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def on_enable(self, msg):
        self.enabled = bool(msg.data)

    # ------------------------------------------------------------------ 50 Hz
    def tick(self):
        t = self.now()
        feet = [self.lookup(f) for f in self.frames["feet"]]
        feet = [f[1][2] for f in feet if f is not None]
        tops = [self.lookup(f) for f in self.frames["body_top"]]
        tops = [f[1][2] for f in tops if f is not None]
        if feet:
            floor = min(feet) - float(self.cfg.get("foot_sole_offset", 0.0))
            top = max(tops) + float(self.cfg.get("body_top_offset", 0.0)) - floor if tops else None
            self.core.on_body(top_above_floor=top, floor_z=floor)
        stale = any(t - self.t_sensor.get(k, -1e9) > 0.5 for k in ("joints", "odom", "camera"))
        ch = None
        if not stale:
            try:
                ch = self.core.decide(t)
            except RuntimeError as e:
                self.get_logger().error(str(e))
        cmd = self.core.command(t)
        if stale or not self.enabled:
            cmd.update(lin_vel_x=0.0, lin_vel_y=0.0, yaw_rate=0.0)
        cmd["lin_vel_x"] = float(np.clip(cmd["lin_vel_x"], -self.max_vx, self.max_vx))
        self.core.last_sent = np.array([cmd["lin_vel_x"], cmd["lin_vel_y"], 0.0, cmd["yaw_rate"]])  # what was really sent
        self.publish(cmd)
        if ch is not None:
            top3 = sorted(ch["table"], key=lambda r: -r["U"])[:3]
            self.pub_status.publish(String(data=json.dumps(dict(
                t=round(t, 2), cls=ch["cls"], model_cls=ch.get("model_cls"), skill=(ch.get("rule") or {}).get("skill"),
                U=round(ch["U"], 4), p=round(ch["p"], 4), cmd=ch["cmd"], enabled=self.enabled,
                top3=[dict(cls=r["cls"], U=r["U"], p_fall=r["p_fall"]) for r in top3]))))

    def publish(self, cmd):
        m = ControllerCommand()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frames["world"]
        m.controller_id = self.cfg["controller_id"]
        m.lin_vel_x, m.lin_vel_y, m.yaw_rate = cmd["lin_vel_x"], cmd["lin_vel_y"], cmd["yaw_rate"]
        f = lambda k: float(cmd[k]) if cmd.get(k) is not None else NAN
        m.body_height, m.torso_pitch, m.torso_roll, m.speed = f("pelvis_height"), f("torso_pitch"), f("torso_roll"), f("speed")
        m.mode = str(cmd.get("mode") or "")
        m.face = float(cmd.get("face") or 0.0)
        m.xfeas_class, m.skill = str(cmd.get("cls") or ""), str(cmd.get("skill") or "")
        m.score = f("U")
        m.enabled, m.at_goal = self.enabled, bool(cmd.get("at_goal"))
        self.pub_cmd.publish(m)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = cmd["lin_vel_x"], cmd["lin_vel_y"], cmd["yaw_rate"]
        self.pub_twist.publish(tw)

    def destroy_node(self):
        if self.core.client is not None:
            self.core.client.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = XFeasNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(dict(lin_vel_x=0.0, lin_vel_y=0.0, yaw_rate=0.0))  # stop on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
