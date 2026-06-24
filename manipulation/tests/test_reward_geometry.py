"""Sanity check: read the hand/apple/bowl poses from the task and print the
hand-to-apple and apple-to-bowl distances used by the shaped reward."""

import gymnasium as gym
import mani_skill.envs
import numpy as np

def main():
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1", obs_mode="state", control_mode="pd_joint_pos")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    unwrapped = env.unwrapped
    try:
        apple_p = unwrapped.apple.pose.p.cpu().numpy()
        bowl_p = unwrapped.bowl.pose.p.cpu().numpy()
        hand_p = unwrapped.agent.right_tcp.pose.p.cpu().numpy()
        
        # Squeeze if needed
        if apple_p.ndim > 1: apple_p = apple_p[0]
        if bowl_p.ndim > 1: bowl_p = bowl_p[0]
        if hand_p.ndim > 1: hand_p = hand_p[0]
        
        dist_hand_apple = np.linalg.norm(hand_p - apple_p)
        dist_apple_bowl = np.linalg.norm(apple_p - bowl_p)
        
        print("Success! Hand to Apple:", dist_hand_apple, "Apple to Bowl:", dist_apple_bowl)
        print("Original reward:", float(reward.cpu().numpy() if hasattr(reward, 'cpu') else reward))
    except Exception as e:
        print("Error accessing positions:", e)

if __name__ == "__main__":
    main()
