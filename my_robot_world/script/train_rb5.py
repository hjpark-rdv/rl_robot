import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import time
import argparse
from datetime import datetime
import glob

class RobotArmEnv(gym.Env):
    def __init__(self, render=False):
        super(RobotArmEnv, self).__init__()
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self._load_urdf_once()

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.joint_indices = [0, 1, 4, 7, 8, 9]
        self.step_counter = 0
        self.max_steps = 1000
        self.target_distance = 0.01
        self.stay_counter = 0
        self.stay_threshold = 120
        self.prev_distance = None
        self.base_pos = np.array([0, 0, 0], dtype=np.float32)  # 로봇 베이스 위치

    def _load_urdf_once(self):
        urdf_path = "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/rb5_moveit.urdf"
        target_path = "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/cube_small.urdf"
        if not os.path.exists(urdf_path) or not os.path.exists(target_path):
            raise FileNotFoundError("URDF files not found")
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        self.target_id = None
        self._find_end_effector_index()

    def _generate_target_position(self, difficulty="medium"):
        if difficulty == "easy":
            r = np.random.uniform(0.5, 0.8)
            z = np.random.uniform(0.4, 0.7)
        elif difficulty == "medium":
            r = np.random.uniform(0.5, 0.8)
            z = np.random.uniform(0.4, 0.8)
        elif difficulty == "hard":
            r = np.random.uniform(0.5, 0.8)
            z = np.random.uniform(0.4, 0.8)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return [x, y, z]

    def reset(self, seed=None, options=None, difficulty="medium"):
        p.setGravity(0, 0, -9.81)
        for idx in self.joint_indices:
            p.resetJointState(self.robot_id, idx, 0)

        if self.target_id is not None:
            p.removeBody(self.target_id)
        target_pos = self._generate_target_position(difficulty)
        self.target_id = p.loadURDF(
            "/home/rdv/docker_share/tomato_robot/ur_gazebo_ppo_robot/src/my_robot_world/urdf/cube_small.urdf",
            basePosition=target_pos,
            globalScaling=1,
            useFixedBase=True
        )
        print(f"Target Position: {target_pos}")

        self.step_counter = 0
        self.stay_counter = 0
        self.prev_distance = None
        return self._get_obs(), {}

    def _find_end_effector_index(self):
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "end_effector":
                self.end_effector_index = i
                print(f"End Effector Index: {self.end_effector_index}")
                return
        raise ValueError("End effector 'end_effector' not found in URDF")

    def _get_obs(self):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = np.array(ee_state[0], dtype=np.float32)
        target_pos = np.array(p.getBasePositionAndOrientation(self.target_id)[0], dtype=np.float32)
        joint_angles = np.array([p.getJointState(self.robot_id, idx)[0] for idx in self.joint_indices], dtype=np.float32)
        obs = np.concatenate([ee_pos, target_pos, joint_angles]).astype(np.float32)
        return obs

    def step(self, action):
        joint_limits = [p.getJointInfo(self.robot_id, idx)[8:10] for idx in self.joint_indices]
        scaled_action = np.array([np.interp(a, [-1, 1], limits) for a, limits in zip(action, joint_limits)])

        for idx, angle in zip(self.joint_indices, scaled_action):
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=angle, force=500)

        p.stepSimulation()
        time.sleep(1./240. if p.getConnectionInfo(self.physics_client)['isConnected'] else 0)

        obs = self._get_obs()
        ee_pos = obs[:3]
        target_pos = obs[3:6]
        distance = np.linalg.norm(ee_pos - target_pos)

        if distance <= self.target_distance:
            self.stay_counter += 1
        else:
            self.stay_counter = 0

        # 방향 벡터 계산
        base_to_target = target_pos - self.base_pos
        base_to_ee = ee_pos - self.base_pos
        norm_base_to_target = np.linalg.norm(base_to_target)
        norm_base_to_ee = np.linalg.norm(base_to_ee)
        if norm_base_to_target > 0 and norm_base_to_ee > 0:
            cos_similarity = np.dot(base_to_target, base_to_ee) / (norm_base_to_target * norm_base_to_ee)
        else:
            cos_similarity = 1.0  # 제로 벡터 방지

        # 개선된 보상 함수
        reward = -distance * 2.0
        distance_diff = 0.0  # 기본값 초기화
        if self.prev_distance is not None:
            distance_diff = self.prev_distance - distance
            reward += 10.0 * distance_diff
            # 오버슈트 페널티
            if distance_diff < 0 and distance > self.target_distance:
                reward -= 20.0
        if distance <= self.target_distance:
            reward += 1.0 * (self.stay_counter / self.stay_threshold)
        if self.stay_counter >= self.stay_threshold:
            reward += 30.0
        # 방향 보상: 타겟 방향과 일치하면 보상, 지나치면 페널티
        reward += 5.0 * cos_similarity  # 방향 일치도 보상
        if cos_similarity < 0:  # 타겟을 지나친 경우
            reward -= 10.0
        reward -= 0.01 * np.sum(np.abs(action))
        reward = float(reward)
        self.prev_distance = distance

        terminated = bool(self.stay_counter >= self.stay_threshold)
        self.step_counter += 1
        truncated = bool(self.step_counter >= self.max_steps)
        info = {
            "distance": distance,
            "stay_counter": self.stay_counter,
            "overshoot": distance_diff < 0 if self.prev_distance else False,
            "cos_similarity": cos_similarity
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect()

def get_latest_tensorboard_run_number(tensorboard_dir):
    if not os.path.exists(tensorboard_dir):
        return 0
    run_dirs = glob.glob(os.path.join(tensorboard_dir, "run_*"))
    if not run_dirs:
        return 0
    run_numbers = [int(d.split('_')[-1]) for d in run_dirs if d.split('_')[-1].isdigit()]
    return max(run_numbers) + 1 if run_numbers else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PPO robot arm")
    parser.add_argument('--mode', type=str, default='multi', choices=['single', 'multi', 'test'],
                        help="Mode: 'single' (1 CPU with GUI), 'multi' (multi CPU without GUI), 'test' (test with GUI)")
    parser.add_argument('--cont', action='store_true',
                        help="Continue training from saved model (multi mode only)")
    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = "./ppo_robot_tensorboard/"
    log_number = get_latest_tensorboard_run_number(tensorboard_dir)
    model_filename = f"ppo_robot_arm_{current_time}_{log_number}"

    if args.mode == "single":
        print("Using single environment with GUI")
        env = RobotArmEnv(render=True)
        check_env(env)

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_dir)
        model.learn(total_timesteps=100000)
        model.save(model_filename)
        print(f"Model saved as {model_filename}.zip")

        obs, _ = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Distance: {info['distance']:.4f}m, Stay: {info['stay_counter']}, Reward: {reward:.4f}")
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    elif args.mode == "multi":
        print("Using vectorized environment (SubprocVecEnv) without GUI")
        num_envs = os.cpu_count() or 4
        env = SubprocVecEnv([lambda: RobotArmEnv(render=False) for _ in range(num_envs)])
        env = VecMonitor(env)
        check_env(RobotArmEnv(render=False))

        eval_env = SubprocVecEnv([lambda: RobotArmEnv(render=False) for _ in range(num_envs)])
        eval_env = VecMonitor(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./logs/',
            log_path='./logs/',
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1
        )

        if args.cont:
            model_path = "ppo_robot_arm.zip"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found for continuing training")
            print("Continuing training from ppo_robot_arm.zip")
            model = PPO.load(model_path, env=env)
            model.learn(total_timesteps=40000000, callback=eval_callback, reset_num_timesteps=False)
            model.save(model_filename)
            print(f"Model saved as {model_filename}.zip")
        else:
            print("Starting new training")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=2,
                tensorboard_log=tensorboard_dir,
                learning_rate=0.001,
                n_steps=4096,
                clip_range=0.1,
                ent_coef=0.01
            )
            model.learn(total_timesteps=40000000, callback=eval_callback)
            model.save(model_filename)
            print(f"Model saved as {model_filename}.zip")

        env.close()
        test_env = RobotArmEnv(render=True)
        obs, _ = test_env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            print(f"Distance: {info['distance']:.4f}m, Stay: {info['stay_counter']}, Reward: {reward:.4f}")
            if terminated or truncated:
                obs, _ = test_env.reset()
        test_env.close()

    elif args.mode == "test":
        print("Testing with pre-trained model - Diverse Scenarios")
        # model_path = f"ppo_robot_arm_{current_time}_{log_number}.zip"
        model_path = f"ppo_robot_arm.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model = PPO.load(model_path)
        test_env = RobotArmEnv(render=True)
        success_count = 0
        total_attempts = 150

        difficulties = ["easy"] * 50 + ["medium"] * 50 + ["hard"] * 50

        for episode in range(total_attempts):
            difficulty = difficulties[episode]
            print(f"\nEpisode {episode + 1}/{total_attempts} (Difficulty: {difficulty})")
            obs, _ = test_env.reset(difficulty=difficulty)
            reached = False

            for step in range(test_env.max_steps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                print(f"Step {step + 1}: Distance: {info['distance']:.4f}m, Stay: {info['stay_counter']}, "
                      f"CosSim: {info['cos_similarity']:.4f}, Reward: {reward:.4f}")
                if terminated:
                    print("Target reached!")
                    reached = True
                    success_count += 1
                    break
                if truncated:
                    print("Failed to reach target within max steps")
                    break

            print(f"Episode Result: {'Success' if reached else 'Failure'}")

        print(f"\nTest Summary: {success_count}/{total_attempts} targets reached ({success_count/total_attempts*100:.2f}%)")
        test_env.close()
