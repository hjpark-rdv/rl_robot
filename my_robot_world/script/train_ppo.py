#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tomato_harvest_env import TomatoHarvestEnv
import rospy

rospy.init_node('ppo_training_node', anonymous=True)
rospy.loginfo("Starting PPO training")

# 환경 생성
env = TomatoHarvestEnv()
rospy.loginfo("Environment created")

# 환경 체크
check_env(env)
rospy.loginfo("Environment checked")

# PPO 모델 설정
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tomato_log/",
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
rospy.loginfo("PPO model initialized")

# 학습 시작
model.learn(total_timesteps=100000)
rospy.loginfo("Training completed")

# 모델 저장
model.save("ppo_tomato_harvest")
rospy.loginfo("Model saved")

# 테스트
state, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(state)
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        state, _ = env.reset()
    rospy.loginfo(f"Reward: {reward}")

env.close()