"""Custom dense reward and evaluation for UnitreeG1PlaceAppleInBowl-v1.

`patch_reward()` monkey-patches ManiSkill's `HumanoidPlaceAppleInBowl` task with
a staged dense reward (reach -> grasp -> place -> release/settle -> success) that
removes the "reward cliff" between carrying and dropping the apple, plus a more
forgiving success check. Call it once before creating the environment.
"""

import torch
from typing import Any
from mani_skill.envs.tasks.humanoid.humanoid_pick_place import HumanoidPlaceAppleInBowl

def patch_reward():
    def custom_compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.right_tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward.clone()

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # encourage to bring apple to above the bowl then drop it.
        obj_to_goal_dist = torch.linalg.norm(
            (self.bowl.pose.p + torch.tensor([0, 0, 0.15], device=self.device))
            - self.apple.pose.p,
            axis=1,
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # once above the goal, encourage to have the hand above the bowl still and begin releasing the grasp
        obj_high_above_bowl = obj_to_goal_dist < 0.025
        grasp_release_reward = self._grasp_release_reward()
        
        # fix: reward the apple falling/resting in the bowl to prevent the reward cliff
        apple_to_bowl_dist = torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1)
        xy_dist = torch.linalg.norm((self.bowl.pose.p - self.apple.pose.p)[:, :2], axis=1)
        z_dist = self.apple.pose.p[:, 2] - self.bowl.pose.p[:, 2]
        apple_above_or_in_bowl = (xy_dist < 0.08) & (z_dist > -0.05) & (z_dist < 0.20)
        
        override_mask = obj_high_above_bowl | apple_above_or_in_bowl
        
        # encourage hand to stay near the bowl during drop
        tcp_to_bowl_dist = torch.linalg.norm(
            (self.bowl.pose.p + torch.tensor([0, 0, 0.2], device=self.device)) - self.agent.right_tcp.pose.p,
            axis=1
        )
        hand_near_bowl_reward = 1 - torch.tanh(5 * tcp_to_bowl_dist)

        reward[override_mask] = (
            4
            + (1 - torch.tanh(5 * apple_to_bowl_dist[override_mask])) * 2
            + grasp_release_reward[override_mask]
            + hand_near_bowl_reward[override_mask]
        )
        
        reward[info["success"]] = (
            50 + (1 - torch.tanh(5 * apple_to_bowl_dist[info["success"]])) * 2 + grasp_release_reward[info["success"]]
        )
        return reward

    def custom_evaluate(self):
        # We increase the tolerance from 0.05 to 0.08 because the Euclidean norm includes Z-distance.
        # Since the apple rests on the bottom of the bowl, its Z-distance is often ~0.04m above the bowl origin.
        # If it is slightly off-center (e.g., xy=0.04), sqrt(0.04^2 + 0.04^2) = 0.056 > 0.05, causing false failures.
        is_obj_placed = (
            torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1) <= 0.08
        )
        hand_outside_bowl = (
            self.agent.right_tcp.pose.p[:, 2] > self.bowl.pose.p[:, 2] + 0.125
        )
        is_grasped = self.agent.right_hand_is_grasping(self.apple, max_angle=110)
        return {
            "success": is_obj_placed & hand_outside_bowl,
            "hand_outside_bowl": hand_outside_bowl,
            "is_grasped": is_grasped,
        }

    HumanoidPlaceAppleInBowl.compute_dense_reward = custom_compute_dense_reward
    HumanoidPlaceAppleInBowl.evaluate = custom_evaluate
    print("ManiSkill Reward Function & Evaluate Function Patched!")
