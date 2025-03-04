#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SpawnModel, SpawnModelRequest, DeleteModel, DeleteModelRequest
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker
import tf

class TomatoHarvestEnv(gym.Env):
    def __init__(self):
        assert rospy.core.is_initialized(), "ROS node must be initialized before creating the environment"
        rospy.loginfo("Initializing TomatoHarvestEnv")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "joint_angles": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
            "tomato_pos": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),
            "depth": spaces.Box(low=0, high=np.inf, shape=(480, 640), dtype=np.float32)
        })

        self.joint_pub = rospy.Publisher('/ur3e/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)
        self.joint_sub = rospy.Subscriber('/ur3e/joint_states', JointState, self.joint_callback)
        self.depth_sub = rospy.Subscriber('/rgbd_camera/depth/image_raw', Image, self.depth_callback)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.marker_pub = rospy.Publisher('/tomato_marker', Marker, queue_size=10)
        self.tf_listener = tf.TransformListener()

        self.joint_angles = np.zeros(6, dtype=np.float32)
        self.depth_image = np.zeros((480, 640), dtype=np.float32)
        self.tomato_pos = np.array([0.5, 0.0, 0.5], dtype=np.float32)
        self.rng = np.random.default_rng()
        self.joint_data_received = False
        self.near_tomato_start_time = None
        self.required_hold_time = 1.0
        self.step_count = 0
        self.max_steps = 100

        self.joint_limits = {
            'min': np.array([-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi], dtype=np.float32),
            'max': np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi], dtype=np.float32)
        }
        self.max_reach = 0.5
        self.min_reach = 0.1

        self.pipe_sdf = """
        <sdf version='1.6'>
            <model name='pipe'>
                <pose>0 0 0.5 0 0 0</pose>
                <link name='pipe_link'>
                    <gravity>0</gravity>
                    <visual name='pipe_visual'>
                        <geometry>
                            <cylinder>
                                <radius>0.0025</radius>
                                <length>1.0</length>
                            </cylinder>
                        </geometry>
                        <material>
                            <diffuse>0 0 1 1</diffuse>
                        </material>
                    </visual>
                    <collision name='pipe_collision'>
                        <geometry>
                            <cylinder>
                                <radius>0.0025</radius>
                                <length>1.0</length>
                            </cylinder>
                        </geometry>
                    </collision>
                    <inertial>
                        <mass>0.1</mass>
                        <inertia>
                            <ixx>0.001</ixx>
                            <ixy>0</ixy>
                            <ixz>0</ixz>
                            <iyy>0.001</iyy>
                            <iyz>0</iyz>
                            <izz>0.001</izz>
                        </inertia>
                    </inertial>
                </link>
            </model>
        </sdf>
        """

        self._reset_robot_position()
        self._reset_tomato_position()
        self._spawn_pipe()

        rospy.loginfo("TomatoHarvestEnv initialized")

    def joint_callback(self, msg):
        try:
            self.joint_angles = np.array(msg.position[:6], dtype=np.float32)
            if np.any(np.isnan(self.joint_angles)):
                self.joint_angles = np.nan_to_num(self.joint_angles, nan=0.0)
            self.joint_data_received = True
        except Exception as e:
            rospy.logerr(f"Error in joint callback: {e}")
            self.joint_angles = np.zeros(6, dtype=np.float32)

    def depth_callback(self, msg):
        try:
            if msg.encoding == "32FC1":
                self.depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
            else:
                self.depth_image = np.zeros((480, 640), dtype=np.float32)
            self.depth_image = np.nan_to_num(self.depth_image, nan=0.0)
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {e}")
            self.depth_image = np.zeros((480, 640), dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        new_joint_angles = self.joint_angles + action * 0.05
        new_joint_angles = np.clip(new_joint_angles, self.joint_limits['min'], self.joint_limits['max'])
        point.positions = new_joint_angles.tolist()
        point.time_from_start = rospy.Duration(0.05)
        msg.points.append(point)
        self.joint_pub.publish(msg)
        rospy.sleep(0.05)

        state = self._get_state()
        ee_pos = self._get_end_effector_pos()
        dist_to_tomato = np.linalg.norm(state["tomato_pos"] - ee_pos)
        if np.isnan(dist_to_tomato):
            dist_to_tomato = 1.0

        # 보상: 토마토 접근만 계산, 파이프는 depth로 간접 감지
        reward = -dist_to_tomato
        # depth 데이터의 변화로 파이프 근접성 간접 평가
        depth_mean = np.mean(state["depth"])
        if depth_mean < 0.5:  # 임의 기준: 가까운 장애물 감지 시 페널티
            reward -= 5.0

        done = False
        truncated = False

        if dist_to_tomato < 0.01:
            if self.near_tomato_start_time is None:
                self.near_tomato_start_time = rospy.Time.now().to_sec()
            time_held = rospy.Time.now().to_sec() - self.near_tomato_start_time
            if time_held >= self.required_hold_time:
                reward += 100.0
                done = True
        else:
            self.near_tomato_start_time = None

        if self.step_count >= self.max_steps:
            truncated = True
            reward -= 50.0

        return state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_robot_position()
        self._reset_tomato_position()
        self._spawn_pipe()

        self.near_tomato_start_time = None
        self.step_count = 0

        rospy.sleep(0.5)
        state = self._get_state()
        return state, {}

    def _reset_robot_position(self):
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, -2.57, 1.57, 0.0, 1.57, 0.0]
        point.time_from_start = rospy.Duration(0.5)
        msg.points.append(point)
        self.joint_pub.publish(msg)

    def _reset_tomato_position(self):
        max_attempts = 10
        for attempt in range(max_attempts):
            x = float(self.rng.uniform(0.3, 0.5))
            y = float(self.rng.uniform(-0.3, 0.3))
            z = float(self.rng.uniform(0.2, 0.5))
            candidate_pos = np.array([x, y, z], dtype=np.float32)
            if self._is_reachable(candidate_pos):
                self.tomato_pos = candidate_pos
                break
        else:
            self.tomato_pos = np.array([0.4, 0.0, 0.3], dtype=np.float32)

        req = SetModelStateRequest()
        req.model_state.model_name = 'tomato'
        req.model_state.pose = Pose(position=Point(self.tomato_pos[0], self.tomato_pos[1], self.tomato_pos[2]),
                                   orientation=Quaternion(0, 0, 0, 1))
        req.model_state.reference_frame = 'world_link'
        try:
            self.set_model_state(req)
        except rospy.ServiceException as e:
            rospy.logerror(f"Failed to update tomato position: {e}")
            raise

        self._publish_marker()

    def _spawn_pipe(self):
        try:
            delete_req = DeleteModelRequest()
            delete_req.model_name = 'pipe'
            self.delete_model(delete_req)
            rospy.sleep(0.05)
        except rospy.ServiceException:
            pass

        # 로봇과 토마토 사이, 토마토와 1cm 이내 위치
        robot_pos = np.array([0.0, 0.0, 0.0])
        direction = self.tomato_pos - robot_pos
        dist_to_tomato = np.linalg.norm(direction)
        if dist_to_tomato > 0:
            direction /= dist_to_tomato

        offset = self.rng.uniform(0.005, 0.01)  # 0.5cm ~ 1cm
        pipe_base_x = self.tomato_pos[0] - (offset*10) * direction[0]
        pipe_base_y = self.tomato_pos[1] - (offset*3) * direction[1]
        pipe_base_z = 0.0  # 바닥에 닿음
        pipe_z = 0.0  # 관 중심 높이

        spawn_req = SpawnModelRequest()
        spawn_req.model_name = 'pipe'
        spawn_req.model_xml = self.pipe_sdf
        spawn_req.initial_pose = Pose(position=Point(pipe_base_x, pipe_base_y, pipe_z),
                                     orientation=Quaternion(0, 0, 0, 1))
        spawn_req.reference_frame = 'world_link'
        try:
            self.spawn_model(spawn_req)
            rospy.loginfo(f"Pipe spawned at: [{pipe_base_x}, {pipe_base_y}, {pipe_z}]")
        except rospy.ServiceException as e:
            rospy.logerror(f"Failed to spawn pipe: {e}")

    def _get_state(self):
        if self.depth_image is None or not self.joint_data_received:
            rospy.sleep(0.01)
        return {
            "joint_angles": self.joint_angles.astype(np.float32),
            "tomato_pos": self.tomato_pos.astype(np.float32),
            "depth": self.depth_image.astype(np.float32)
        }

    def _get_end_effector_pos(self):
        try:
            self.tf_listener.waitForTransform("/world_link", "/end_effector", rospy.Time(0), rospy.Duration(0.1))
            (trans, _) = self.tf_listener.lookupTransform("/world_link", "/end_effector", rospy.Time(0))
            return np.array(trans, dtype=np.float32)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return np.array([0.5, 0.0, 0.5], dtype=np.float32)

    def _publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "world_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "tomato"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.tomato_pos[0])
        marker.pose.position.y = float(self.tomato_pos[1])
        marker.pose.position.z = float(self.tomato_pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)

    def _is_reachable(self, pos):
        dist_from_base = np.linalg.norm(pos)
        return self.min_reach <= dist_from_base <= self.max_reach

if __name__ == '__main__':
    rospy.init_node('test_node', anonymous=True)
    env = TomatoHarvestEnv()
    state = env.reset()
    for _ in range(2000):
        action = env.action_space.sample()
        state, reward, done, truncated, _ = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
        if done or truncated:
            state = env.reset()