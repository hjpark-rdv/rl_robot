import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

class CustomRobotEnv(gym.Env):
    def __init__(self, urdf_path, render_mode="human"):
        super(CustomRobotEnv, self).__init__()

        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.urdf_path = urdf_path
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

        self.joint_indices = [0, 1, 4, 7, 8, 9]
        self.end_effector_index = 12
        self.initial_joint_positions = [0.0] * len(self.joint_indices)

        self.target_id = None
        self.obstacle_ids = []

        self.step_count = 0
        self.max_steps = 1000

        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def create_target(self):
        target_pos = np.random.uniform([-0.4, -0.4, 0.3], [0.4, 0.4, 1.1]).astype(np.float32)
        target_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=target_shape,
            basePosition=target_pos,
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        )
        return target_pos

    def create_obstacles(self, target_pos):
        self.obstacle_ids = []
        for _ in range(3):
            offset = np.random.uniform([-0.15, -0.15, -0.15], [0.15, 0.15, 0.15]).astype(np.float32)
            obstacle_pos = target_pos + offset
            obstacle_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_shape,
                basePosition=obstacle_pos,
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
            )
            self.obstacle_ids.append(obstacle_id)
        return [p.getBasePositionAndOrientation(obs_id)[0] for obs_id in self.obstacle_ids]

    def reset(self, seed=None, options=None):
        for idx, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, self.initial_joint_positions[idx])

        if self.target_id is not None:
            p.removeBody(self.target_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.target_id = None
        self.obstacle_ids = []

        self.target_pos = self.create_target()
        self.obstacle_positions = self.create_obstacles(self.target_pos)

        self.step_count = 0
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        observation = self._get_observation(np.array(ee_state[0], dtype=np.float32))
        return observation, {}

    def _get_observation(self, current_pos):
        distances = [np.linalg.norm(current_pos - obs) for obs in self.obstacle_positions]
        distances = distances + [1.0] * (3 - len(distances)) if len(distances) < 3 else distances[:3]
        return np.concatenate([current_pos, self.target_pos, distances], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        current_pos = np.array(ee_state[0], dtype=np.float32)

        target_pos = current_pos + action
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            lowerLimits=[-np.pi] * len(self.joint_indices),
            upperLimits=[np.pi] * len(self.joint_indices),
            jointRanges=[2 * np.pi] * len(self.joint_indices),
            restPoses=[0] * len(self.joint_indices),
            maxNumIterations=100,
            residualThreshold=0.01
        )

        if not joint_angles or len(joint_angles) < len(self.joint_indices):
            reward = -50
            observation = self._get_observation(current_pos)
            terminated = False
            truncated = True
            info = {"reason": "IK unreachable"}
            return observation, reward, terminated, truncated, info

        for idx, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[idx]
            )

        p.stepSimulation()
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        new_pos = np.array(ee_state[0], dtype=np.float32)
        observation = self._get_observation(new_pos)

        distance_to_target = np.linalg.norm(new_pos - self.target_pos)
        min_obstacle_distance = min([np.linalg.norm(new_pos - obs) for obs in self.obstacle_positions])
        
        reward = -distance_to_target
        if min_obstacle_distance < 0.06:
            reward -= 10
        if distance_to_target < 0.05:
            reward += 100

        terminated = bool(distance_to_target < 0.05)
        truncated = bool(min_obstacle_distance < 0.06 or self.step_count >= self.max_steps)
        info = {"joint_angles": joint_angles}

        if self.step_count >= self.max_steps:
            info["reason"] = "Max steps exceeded"

        self.step_count = 0 if terminated or truncated else self.step_count
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect()

def make_env(urdf_path, render_mode="direct"):
    def _init():
        return CustomRobotEnv(urdf_path, render_mode=render_mode)
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

def train_model(env, mode, total_timesteps=5000000, load_path=None):
    if load_path and os.path.exists(load_path):
        print(f"Loading existing model from {load_path} for continued training...")
        model = PPO.load(load_path, env=env, tensorboard_log="./tensorboard_logs/", device="cpu")
    else:
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, tensorboard_log="./tensorboard_logs/", device="cpu")
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./checkpoints/",
                                             name_prefix=f"ppo_robot_{mode}")
    reward_callback = RewardPlotCallback()
    # model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, reward_callback], log_interval=10)
    model.learn(total_timesteps=total_timesteps)
    
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
        print(f"Step {step + 1}: Position = {observation[:3]}, Reward = {reward}")
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
        env = CustomRobotEnv(urdf_path, render_mode="human")
        check_env(env)
        model = train_model(env, "single")
        test_model(model, env)

    elif args.mode == "multi":
        num_cpu = 40
        env = SubprocVecEnv([make_env(urdf_path, render_mode="direct") for _ in range(num_cpu)])
        model = train_model(env, "multi")
        test_env = CustomRobotEnv(urdf_path, render_mode="human")
        timestamp = get_current_timestamp()
        model = PPO.load(f"weight/ppo_robot_obstacle_avoidance_multi_{timestamp}")
        test_model(model, test_env)
        print(f"weight/ppo_robot_obstacle_avoidance_multi_{timestamp}")

    elif args.mode == "continue":
        load_path = "weight/ppo_robot_obstacle_avoidance.zip"
        env = CustomRobotEnv(urdf_path, render_mode="human")
        check_env(env)
        model = train_model(env, "continue", load_path=load_path)
        test_model(model, env)

    elif args.mode == "test":
        # Test 모드: 저장된 모델 로드 후 GUI로 테스트
        env = CustomRobotEnv(urdf_path, render_mode="human")
        # 최신 모델을 찾기 위해 weight 폴더에서 파일 목록 확인
        # weight_files = [f for f in os.listdir("weight") if f.startswith("ppo_robot_obstacle_avoidance_") and f.endswith(".zip")]
        weight_files = "ppo_robot_obstacle_avoidance.zip"
        # if not weight_files:
            # raise FileNotFoundError("No trained model found in 'weight' folder")
        # latest_model = max(weight_files, key=lambda x: os.path.getmtime(os.path.join("weight", x)))
        # model_path = os.path.join("weight", latest_model)
        model_path = os.path.join("./", weight_files)
        print(f"Loading model from {model_path} for testing...")
        model = PPO.load(model_path)
        test_model(model, env)

# TensorBoard 실행: tensorboard --logdir ./tensorboard_logs/