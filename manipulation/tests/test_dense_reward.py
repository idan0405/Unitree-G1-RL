"""Sanity check: confirm the place-apple task returns a dense (not sparse)
reward by stepping once with a random action and inspecting the value."""

import gymnasium as gym
import mani_skill.envs

def main():
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1", obs_mode="state", control_mode="pd_joint_pos")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward type: {type(reward)}, Value: {reward}")
    if isinstance(reward, float) and reward != 0.0 and reward != -1.0:
        print("Looks like a dense reward!")
    else:
        print("Might be sparse. Check info dict:", info.keys())

if __name__ == "__main__":
    main()
