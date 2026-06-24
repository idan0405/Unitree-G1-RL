"""Sanity check: wrap the place-apple task in ManiSkill's SB3 vector wrapper and
confirm reset/step return the shapes Stable-Baselines3 expects."""

import gymnasium as gym
import mani_skill.envs
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from stable_baselines3 import PPO

def main():
    print("Creating env...")
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1", num_envs=256, obs_mode="state", control_mode="pd_joint_pos", max_episode_steps=200)
    
    if not hasattr(env, "num_envs"):
        env.num_envs = env.unwrapped.num_envs
        env.single_observation_space = env.unwrapped.single_observation_space
        env.single_action_space = env.unwrapped.single_action_space
        
    vec_env = ManiSkillSB3VectorEnv(env)
    
    print("Testing step...")
    obs = vec_env.reset()
    print("Obs shape:", obs.shape, "Type:", type(obs))
    
    action = vec_env.action_space.sample()
    print("Action shape:", action.shape, "Type:", type(action))
    
    obs, reward, done, info = vec_env.step(action)
    print("Step success!")
    print("Reward shape:", reward.shape, "Type:", type(reward))
    print("Done shape:", done.shape, "Type:", type(done))
    
if __name__ == "__main__":
    main()
