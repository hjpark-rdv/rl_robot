import gymnasium as gym
from gymnasium import spaces
import rospy
import numpy as np
import time
import argparse
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import moveit_commander
import pybullet as p
import pybullet_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import tf

class CustomRobotEnv(gym.Env):
    def __init__(self, engine="gazebo", render_mode="human"):
        super(CustomRobotEnv, self).__init__()
        self.engine = engine.lower()
        self.render_mode = render_mode

        # joint_names 먼저 정의
        self.joint_names = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
        self.initial_joint_positions = [0.0] * len(self.joint_names)  # Gazebo용
        self.initial_joint_positions_pybullet = [0.0, 0.5, -0.921, 1.95, 0, 0.0]  # PyBullet용
        self.target_pos = None
        self.obstacle_positions = []
        self.target_name = "target"
        self.obstacle_names = ["obstacle1", "obstacle2", "obstacle3"]
        self.step_count = 0
        self.max_steps = 1000
        self.target_id = None
        self.obstacle_ids = []

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.target_sdf = """
        <sdf version="1.6">
          <model name="target">
            <static>true</static>
            <link name="base_link">
              <visual name="visual">
                <geometry>
                  <sphere><radius>0.01</radius></sphere>
                </geometry>
                <material>
                  <diffuse>1 0 0 1</diffuse>
                </material>
              </visual>
              <collision name="collision">
                <geometry>
                  <sphere><radius>0.01</radius></sphere>
                </geometry>
              </collision>
            </link>
          </model>
        </sdf>
        """
        self.obstacle_sdf = """
        <sdf version="1.6">
          <model name="obstacle">
            <static>true</static>
            <link name="base_link">
              <visual name="visual">
                <geometry>
                  <sphere><radius>0.01</radius></sphere>
                </geometry>
                <material>
                  <diffuse>0 0 1 1</diffuse>
                </material>
              </visual>
              <collision name="collision">
                <geometry>
                  <sphere><radius>0.01</radius></sphere>
                </geometry>
              </collision>
            </link>
          </model>
        </sdf>
        """

        # 엔진 초기화는 속성 정의 후 호출
        if self.engine == "gazebo":
            self._init_gazebo()
        elif self.engine == "pybullet":
            self._init_pybullet()

        self.reset()

    def _init_gazebo(self):
        if not rospy.core.is_initialized():
            rospy.init_node("robot_rl_env", anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("RB5")
        self.move_group.set_planner_id("RRTConnect")
        self.move_group.set_end_effector_link("end_effector")
        self.tf_listener = tf.TransformListener()
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.joint_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.ee_pose = None
        rospy.loginfo("Waiting for joint states...")
        timeout = 20.0
        start_time = time.time()
        rate = rospy.Rate(10)
        while self.ee_pose is None and not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            rate.sleep()
        if self.ee_pose is None:
            raise RuntimeError("Failed to receive joint states within 20 seconds.")

    def _init_pybullet(self):
        self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        urdf_path = "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/rb5.urdf"
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"PyBullet URDF file not found at {urdf_path}")
        print(f"Loading URDF from {urdf_path}")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        print(f"Robot ID: {self.robot_id}")
        self.target_id = None
        self.obstacle_ids = []
        self.joint_indices = self._get_joint_indices_from_urdf()
        self._find_end_effector_index()

    def _get_joint_indices_from_urdf(self):
        joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_names:
                joint_indices.append(i)
                print(f"Found joint '{joint_name}' at index {i}")
        if len(joint_indices) != len(self.joint_names):
            raise ValueError(f"Could not find all joints in URDF. Found: {joint_indices}, Expected: {self.joint_names}")
        return joint_indices

    def _find_end_effector_index(self):
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "end_effector":
                self.end_effector_index = i
                return
        raise ValueError("End effector 'end_effector' not found in URDF")

    def joint_callback(self, msg):
        self.ee_pose = msg.position[:6]

    def create_target(self):
        target_pos = np.random.uniform([-0.4, -0.4, 0.2], [0.4, 0.4, 0.6]).astype(np.float32)
        if self.engine == "gazebo":
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = target_pos
            try:
                self.spawn_model(self.target_name, self.target_sdf, "", pose, "world")
                time.sleep(0.5)
                rospy.loginfo(f"Spawned target (red) at {target_pos}")
            except Exception as e:
                rospy.logwarn(f"Failed to spawn target: {e}")
        elif self.engine == "pybullet":
            self.target_id = p.loadURDF(
                "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/cube_small.urdf",
                basePosition=target_pos,
                globalScaling=1,
                useFixedBase=True
            )
        return target_pos

    def create_obstacles(self, target_pos):
        obstacle_positions = []
        self.obstacle_ids = []
        for i, name in enumerate(self.obstacle_names):
            offset = np.random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]).astype(np.float32)
            obstacle_pos = target_pos + offset
            if self.engine == "gazebo":
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = obstacle_pos
                try:
                    self.spawn_model(name, self.obstacle_sdf, "", pose, "world")
                    time.sleep(0.5)
                    rospy.loginfo(f"Spawned obstacle {name} (blue) at {obstacle_pos}")
                    obstacle_positions.append(obstacle_pos)
                except Exception as e:
                    rospy.logwarn(f"Failed to spawn obstacle {name}: {e}")
            elif self.engine == "pybullet":
                obstacle_id = p.loadURDF(
                    "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/collision_small.urdf",
                    basePosition=obstacle_pos,
                    globalScaling=1,
                    useFixedBase=True
                )
                self.obstacle_ids.append(obstacle_id)
                obstacle_positions.append(obstacle_pos)
        return obstacle_positions

    def reset(self, seed=None, options=None):
        if self.engine == "gazebo":
            for name in [self.target_name] + self.obstacle_names:
                try:
                    self.delete_model(name)
                    time.sleep(0.5)
                    rospy.loginfo(f"Deleted {name}")
                except:
                    pass
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            point.positions = self.initial_joint_positions
            point.time_from_start = rospy.Duration(1.0)
            traj.points.append(point)
            self.joint_pub.publish(traj)
            rospy.sleep(2.0)
        elif self.engine == "pybullet":
            p.setGravity(0, 0, -9.81)
            for idx, angle in zip(self.joint_indices, self.initial_joint_positions_pybullet):
                p.resetJointState(self.robot_id, idx, angle)
            if self.target_id is not None:
                p.removeBody(self.target_id)
                self.target_id = None
            for obstacle_id in self.obstacle_ids:
                p.removeBody(obstacle_id)
            self.obstacle_ids = []
            p.stepSimulation()

        self.target_pos = self.create_target()
        self.obstacle_positions = self.create_obstacles(self.target_pos)

        observation = self._get_observation()
        if observation is None:
            raise RuntimeError("Failed to get observation during reset")
        
        self.step_count = 0
        return observation, {}

    def _get_end_effector_pos(self):
        if self.engine == "gazebo":
            try:
                self.tf_listener.waitForTransform("/world", "/end_effector", rospy.Time(0), rospy.Duration(0.1))
                (trans, _) = self.tf_listener.lookupTransform("/world", "/end_effector", rospy.Time(0))
                return np.array(trans, dtype=np.float32)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed: {e}")
                return np.array([0.5, 0.0, 0.5], dtype=np.float32)
        elif self.engine == "pybullet":
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
            return np.array(ee_state[0], dtype=np.float32)

    def _get_observation(self):
        if self.engine == "gazebo":
            if self.ee_pose is None:
                rospy.logwarn("Joint states not available")
                return np.zeros(12, dtype=np.float32)
            current_joints = np.array(self.ee_pose, dtype=np.float32)
        elif self.engine == "pybullet":
            current_joints = np.array([p.getJointState(self.robot_id, idx)[0] for idx in self.joint_indices], dtype=np.float32)
        
        ee_pos = self._get_end_effector_pos()
        
        if self.target_pos is None or self.obstacle_positions is None:
            print("Target or obstacles not initialized")
            return np.zeros(12, dtype=np.float32)
        
        distances = [np.linalg.norm(ee_pos - obs) for obs in self.obstacle_positions]
        distances = distances + [1.0] * (3 - len(distances)) if len(distances) < 3 else distances[:3]
        return np.concatenate([current_joints, self.target_pos, np.array(distances, dtype=np.float32)], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        current_ee_pos = self._get_end_effector_pos()
        delta_pos = action * 0.05
        target_ee_pos = current_ee_pos + delta_pos

        if self.engine == "gazebo":
            if self.ee_pose is None:
                rospy.logwarn("No joint states available during step")
                observation = self._get_observation()
                return observation, -50, False, True, {"reason": "No joint states"}
            self.move_group.set_position_target(target_ee_pos.tolist())
            success = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            if not success:
                rospy.logwarn("MoveIt failed to plan or execute")
                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                point = JointTrajectoryPoint()
                point.positions = self.initial_joint_positions
                point.time_from_start = rospy.Duration(1.0)
                traj.points.append(point)
                self.joint_pub.publish(traj)
                rospy.sleep(1.0)
            rospy.sleep(1.0)
        elif self.engine == "pybullet":
            joint_angles = p.calculateInverseKinematics(self.robot_id, self.end_effector_index, target_ee_pos)
            for idx, angle in zip(self.joint_indices, joint_angles[:6]):
                p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=angle, force=500)
            p.stepSimulation()
            time.sleep(1./240. if self.render_mode == "human" else 0)

        observation = self._get_observation()
        if observation is None:
            return np.zeros(12, dtype=np.float32), -50, False, True, {"reason": "Observation failed"}

        ee_pos = self._get_end_effector_pos()
        distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
        min_obstacle_distance = min([np.linalg.norm(ee_pos - obs) for obs in self.obstacle_positions])
        
        reward = float(-distance_to_target)
        if min_obstacle_distance < 0.06:
            reward -= 10
        if distance_to_target < 0.05:
            reward += 100

        terminated = bool(distance_to_target < 0.05)
        truncated = bool(self.step_count >= self.max_steps or min_obstacle_distance < 0.06)
        info = {"ee_pos": ee_pos, "target_pos": self.target_pos}

        if self.step_count >= self.max_steps:
            info["reason"] = "Max steps exceeded"

        self.step_count = 0 if terminated or truncated else self.step_count
        return observation, reward, terminated, truncated, info

    def close(self):
        if self.engine == "gazebo":
            moveit_commander.roscpp_shutdown()
            rospy.signal_shutdown("Environment closed")
        elif self.engine == "pybullet":
            p.disconnect()

def make_env(urdf_path, engine="gazebo", render_mode="direct"):
    def _init():
        return CustomRobotEnv(engine=engine, render_mode=render_mode)
    return _init

def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPlotCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_means = []
        self.episode_nums = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Real-Time Episode Reward")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Mean Reward")
        self.line, = self.ax.plot([], [], 'b-')

    def _on_step(self):
        reward = np.mean(self.locals["rewards"])
        done = self.locals["dones"][0]
        self.episode_rewards.append(reward)
        if done:
            ep_mean = np.mean(self.episode_rewards)
            if len(self.episode_rewards) >= 1000:
                print("Target potentially unreachable, resetting...")
                ep_mean = -100
            self.episode_means.append(ep_mean)
            self.episode_nums.append(len(self.episode_nums) + 1)
            self.episode_rewards = []
            self.line.set_data(self.episode_nums, self.episode_means)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        return True

    def _on_training_end(self):
        plt.ioff()
        plt.show()

def train_model(env, mode, engine, total_timesteps=200000, load_path=None):
    if load_path and os.path.exists(load_path):
        print(f"Loading existing model from {load_path} for continued training...")
        model = PPO.load(load_path, env=env, tensorboard_log="./tensorboard_logs/", device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            tensorboard_log="./tensorboard_logs/",
            device="cpu",
            learning_rate=0.0003,
            batch_size=64,
            ent_coef=0.01
        )
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/",
                                             name_prefix=f"ppo_robot_{mode}_{engine}")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    timestamp = get_current_timestamp()
    save_path = f"weight/ppo_robot_obstacle_avoidance_{mode}_{engine}_{timestamp}"
    os.makedirs("weight", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model

def test_model(model, env, max_steps=5000):
    observation, info = env.reset()
    print(f"Initial Target Position: {env.target_pos}")
    print(f"Initial Obstacle Positions: {env.obstacle_positions}")

    for step in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: Joints = {observation[:6]}, Reward = {reward}, EE Pos = {info['ee_pos']}")
        if terminated:
            print(f"Reached target position {env.target_pos}")
            observation, info = env.reset()
            print(f"New Target Position: {env.target_pos}")
            print(f"New Obstacle Positions: {env.obstacle_positions}")
        elif truncated:
            print(f"Resetting due to: {info.get('reason', 'unknown')}")
            observation, info = env.reset()
            print(f"New Target Position: {env.target_pos}")
            print(f"New Obstacle Positions: {env.obstacle_positions}")
        time.sleep(1./240. if env.render_mode == "human" else 0)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot RL Training Mode")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi", "continue", "test"],
                        help="Training mode: 'single' (GUI), 'multi' (no GUI, multi-CPU), 'continue' (load and continue), 'test' (GUI test)")
    parser.add_argument("--engine", type=str, default="gazebo", choices=["gazebo", "pybullet"],
                        help="Simulation engine: 'gazebo' or 'pybullet'")
    args = parser.parse_args()

    urdf_path = "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/rb5.urdf"

    if args.mode == "single":
        env = CustomRobotEnv(engine=args.engine, render_mode="human")
        check_env(env)
        model = train_model(env, "single", args.engine, total_timesteps=200000)
        test_model(model, env)

    elif args.mode == "multi":
        num_cpu = 40
        env = SubprocVecEnv([make_env(urdf_path, engine=args.engine, render_mode="direct") for _ in range(num_cpu)])
        env = VecMonitor(env)
        model = train_model(env, "multi", args.engine, total_timesteps=20000000)
        test_env = CustomRobotEnv(engine=args.engine, render_mode="human")
        timestamp = get_current_timestamp()
        model = PPO.load(f"weight/ppo_robot_obstacle_avoidance_multi_{args.engine}_{timestamp}")
        test_model(model, test_env)

    elif args.mode == "continue":
        load_path = f"weight/ppo_robot_obstacle_avoidance_{args.engine}.zip"
        env = CustomRobotEnv(engine=args.engine, render_mode="human")
        check_env(env)
        model = train_model(env, "continue", args.engine, total_timesteps=200000, load_path=load_path)
        test_model(model, env)

    elif args.mode == "test":
        env = CustomRobotEnv(engine=args.engine, render_mode="human")
        latest_model = "ppo_robot_obstacle_avoidance.zip"
        model_path = os.path.join("./", latest_model)
        print(f"Loading model from {model_path} for testing...")
        model = PPO.load(model_path)
        test_model(model, env)