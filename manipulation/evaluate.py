"""
Evaluation script for CleanRL PPO checkpoints.
Loads a .pt state dict and runs the agent in render mode.
"""

import os
import glob
import re

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train import Agent, ENV_ID, SIM_CONFIG


def find_latest_checkpoint(base_dir="runs"):
    """Find the latest checkpoint across all run directories."""
    # Find all run directories
    run_dirs = glob.glob(os.path.join(base_dir, f"{ENV_ID}__*"))
    if not run_dirs:
        print(f"No run directories found in {base_dir}/")
        return None

    latest_dir = max(run_dirs, key=os.path.getmtime)

    # Prefer final checkpoint
    final_path = os.path.join(latest_dir, "final_ckpt.pt")
    if os.path.exists(final_path):
        return final_path

    # Otherwise find the latest numbered checkpoint
    ckpts = glob.glob(os.path.join(latest_dir, "ckpt_*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {latest_dir}/")
        return None

    def get_iter(path):
        match = re.search(r"ckpt_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1

    return max(ckpts, key=get_iter)


def main():
    checkpoint = find_latest_checkpoint()
    if checkpoint is None:
        print("No checkpoint found. Run train.py first.")
        return

    print(f"Loading checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment — single env with human rendering
    env_kwargs = dict(
        obs_mode="state",
        render_mode="human",
        sim_config=SIM_CONFIG,
    )
    import custom_reward
    custom_reward.patch_reward()
    env = gym.make(ENV_ID, num_envs=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    # Load agent
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device))
    agent.eval()

    action_space_low = torch.from_numpy(env.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(env.single_action_space.high).to(device)

    # Evaluation loop
    obs, _ = env.reset()
    total_episodes = 0
    total_successes = 0
    steps = 0

    print("\nRunning evaluation... Press Ctrl+C to stop.\n")
    try:
        while True:
            with torch.no_grad():
                obs = obs.to(device)
                action = agent.get_action(obs, deterministic=True)
                action = torch.clamp(action, action_space_low, action_space_high)

            obs, rewards, terminations, truncations, infos = env.step(action)
            steps += 1
            env.render()
            
            import time
            time.sleep(0.03)  # Slow down rendering to ~30 FPS for visibility

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                if done_mask.any():
                    total_episodes += int(done_mask.sum())
                    
                    # 'success' is stored directly in final_info
                    if "success" in infos["final_info"]:
                        total_successes += int(infos["final_info"]["success"][done_mask].sum())
                    
                    rate = total_successes / total_episodes if total_episodes > 0 else 0
                        
                    print(f"Episode {total_episodes}: success={total_successes}/{total_episodes} "
                          f"({rate:.1%}) | steps={steps}")
                    steps = 0

    except KeyboardInterrupt:
        rate = total_successes / max(1, total_episodes)
        print(f"\nFinal: {total_successes}/{total_episodes} successes ({rate:.1%})")

    env.close()


if __name__ == "__main__":
    main()
