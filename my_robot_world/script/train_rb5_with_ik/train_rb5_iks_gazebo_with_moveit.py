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
    def __init__(self, render_mode="human"):
        super(CustomRobotEnv, self).__init__()

        # ROS 노드 초기화
        rospy.init_node("robot_rl_env", anonymous=True)

        # MoveIt 초기화
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("RB5")
        self.move_group.set_planner_id("RRTConnect")
        self.move_group.set_end_effector_link("end_effector")  # 엔드 이펙터 명시
        self.tf_listener = tf.TransformListener()

        # Gazebo 서비스
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # ROS 토픽
        self.joint_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.ee_pose = None

        # 조인트 상태 대기
        rospy.loginfo("Waiting for joint states...")
        timeout = 20.0
        start_time = time.time()
        rate = rospy.Rate(10)
        while self.ee_pose is None and not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            rate.sleep()
        if self.ee_pose is None:
            raise RuntimeError("Failed to receive joint states within 20 seconds.")

        # 조인트 설정
        self.joint_names = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
        self.initial_joint_positions = [0.0] * len(self.joint_names)

        # 목표와 장애물
        self.target_pos = None
        self.obstacle_positions = []
        self.target_name = "target"
        self.obstacle_names = ["obstacle1", "obstacle2", "obstacle3"]

        self.step_count = 0
        self.max_steps = 1000

        # 행동/관찰 공간
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # 엔드 이펙터 위치 변화량
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # SDF 모델
        self.sphere_sdf = """
        <sdf version="1.6">
          <model name="sphere">
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
        self.obstacle_sdf = self.sphere_sdf.replace("0.02", "0.05").replace("1 0 0 1", "0 0 1 1")

        self.reset()

    def joint_callback(self, msg):
        self.ee_pose = msg.position[:6]

    def create_target(self):
        target_pos = np.random.uniform([0.2, -0.2, 0.3], [0.4, 0.2, 0.6]).astype(np.float32)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = target_pos
        try:
            self.spawn_model(self.target_name, self.sphere_sdf, "", pose, "world")
            time.sleep(0.5)
            rospy.loginfo(f"Spawned target at {target_pos}")
        except Exception as e:
            rospy.logwarn(f"Failed to spawn target: {e}")
        return target_pos

    def create_obstacles(self, target_pos):
        obstacle_positions = []
        for i, name in enumerate(self.obstacle_names):
            offset = np.random.uniform([0.1, 0.1, 0.1], [0.2, 0.2, 0.2]).astype(np.float32)
            obstacle_pos = target_pos + offset
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = obstacle_pos
            try:
                self.spawn_model(name, self.obstacle_sdf, "", pose, "world")
                time.sleep(0.5)
                rospy.loginfo(f"Spawned obstacle {name} at {obstacle_pos}")
                obstacle_positions.append(obstacle_pos)
            except Exception as e:
                rospy.logwarn(f"Failed to spawn obstacle {name}: {e}")
        return obstacle_positions

    def reset(self, seed=None, options=None):
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
        
        self.target_pos = self.create_target()
        self.obstacle_positions = self.create_obstacles(self.target_pos)

        observation = self._get_observation()
        if observation is None:
            raise RuntimeError("Failed to get observation during reset")
        
        self.step_count = 0
        return observation, {}

    def _get_end_effector_pos(self):
        try:
            self.tf_listener.waitForTransform("/world", "/end_effector", rospy.Time(0), rospy.Duration(0.1))
            (trans, _) = self.tf_listener.lookupTransform("/world", "/end_effector", rospy.Time(0))
            return np.array(trans, dtype=np.float32)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return np.array([0.5, 0.0, 0.5], dtype=np.float32)

    def _get_observation(self):
        if self.ee_pose is None:
            rospy.logwarn("Joint states not available")
            return np.zeros(12, dtype=np.float32)
        
        current_joints = np.array(self.ee_pose, dtype=np.float32)
        ee_pos = self._get_end_effector_pos()
        
        if self.target_pos is None or self.obstacle_positions is None:
            rospy.logwarn("Target or obstacles not initialized")
            return np.zeros(12, dtype=np.float32)
        
        distances = [np.linalg.norm(ee_pos - obs) for obs in self.obstacle_positions]
        distances = distances + [1.0] * (3 - len(distances)) if len(distances) < 3 else distances[:3]
        return np.concatenate([current_joints, self.target_pos, np.array(distances, dtype=np.float32)], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        if self.ee_pose is None:
            rospy.logwarn("No joint states available during step")
            observation = self._get_observation()
            return observation, -50, False, True, {"reason": "No joint states"}

        # 현재 엔드 이펙터 위치
        current_ee_pos = self._get_end_effector_pos()
        
        # PPO 행동: 엔드 이펙터 위치 변화량
        delta_pos = action * 0.05  # 스케일링
        target_ee_pos = current_ee_pos + delta_pos

        # MoveIt으로 IK 계산 및 실행
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
        observation = self._get_observation()
        if observation is None:
            return np.zeros(12, dtype=np.float32), -50, False, True, {"reason": "Observation failed"}

        ee_pos = self._get_end_effector_pos()
        distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
        min_obstacle_distance = min([np.linalg.norm(ee_pos - obs) for obs in self.obstacle_positions])
        
        # 보상 계산
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
        print(f"Step {self.step_count}: EE Pos = {ee_pos}, Target = {self.target_pos}, Distance = {distance_to_target}")
        return observation, reward, terminated, truncated, info

    def close(self):
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("Environment closed")

def make_env(urdf_path, render_mode="direct"):
    def _init():
        return CustomRobotEnv(render_mode=render_mode)
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

def train_model(env, mode, total_timesteps=200000, load_path=None):
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
            learning_rate=0.0003,  # 학습 속도 조정
            batch_size=64,         # 안정성 향상
            ent_coef=0.01          # 탐색 강화
        )
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/",
                                             name_prefix=f"ppo_robot_{mode}")
    reward_callback = RewardPlotCallback()
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, reward_callback], log_interval=10)
    
    timestamp = get_current_timestamp()
    save_path = f"weight/ppo_robot_obstacle_avoidance_{mode}_{timestamp}"
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
        time.sleep(1./240.)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot RL Training Mode")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi", "continue", "test"],
                        help="Training mode: 'single' (GUI), 'multi' (no GUI, multi-CPU), 'continue' (load and continue), 'test' (GUI test)")
    args = parser.parse_args()

    urdf_path = "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/rb5_moveit_ee2.urdf"

    if args.mode == "single":
        env = CustomRobotEnv(render_mode="human")
        check_env(env)
        model = train_model(env, "single", total_timesteps=200000)
        test_model(model, env)

    elif args.mode == "multi":
        num_cpu = 4
        env = SubprocVecEnv([make_env(urdf_path, render_mode="direct") for _ in range(num_cpu)])
        env = VecMonitor(env)  # 멀티 환경 모니터링
        model = train_model(env, "multi", total_timesteps=1000000)  # 타임스텝 증가
        test_env = CustomRobotEnv(render_mode="human")
        timestamp = get_current_timestamp()
        model = PPO.load(f"weight/ppo_robot_obstacle_avoidance_multi_{timestamp}")
        test_model(model, test_env)

    elif args.mode == "continue":
        load_path = "weight/ppo_robot_obstacle_avoidance.zip"
        env = CustomRobotEnv(render_mode="human")
        check_env(env)
        model = train_model(env, "continue", total_timesteps=200000, load_path=load_path)
        test_model(model, env)

    elif args.mode == "test":
        env = CustomRobotEnv(render_mode="human")
        weight_files = [f for f in os.listdir("weight") if f.startswith("ppo_robot_obstacle_avoidance_") and f.endswith(".zip")]
        if not weight_files:
            raise FileNotFoundError("No trained model found in 'weight' folder")
        latest_model = max(weight_files, key=lambda x: os.path.getmtime(os.path.join("weight", x)))
        model_path = os.path.join("weight", latest_model)
        print(f"Loading model from {model_path} for testing...")
        model = PPO.load(model_path)
        test_model(model, env)