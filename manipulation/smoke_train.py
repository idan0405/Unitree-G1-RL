"""
CleanRL PPO for UnitreeG1PlaceAppleInBowl-v1
Adapted from the official ManiSkill3 baseline: examples/baselines/ppo/ppo.py

Key differences from the SB3 version:
1. Proper truncation bootstrapping via final_observation
2. Everything stays on GPU (no CPU-GPU transfers)
3. Uses ManiSkillVectorEnv (not SB3's wrapper)
4. finite_horizon_gae for correct advantage estimation
5. Per-minibatch advantage normalization (not VecNormalize)
"""

from collections import defaultdict
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# ===========================================================================
# Configuration — hardcoded for UnitreeG1PlaceAppleInBowl-v1
# Matches official ManiSkill3 defaults from ppo.py
# ===========================================================================
ENV_ID = "UnitreeG1PlaceAppleInBowl-v1"
TOTAL_TIMESTEPS = 12_800 * 2
LEARNING_RATE = 3e-4
NUM_ENVS = 256
NUM_EVAL_ENVS = 4
NUM_STEPS = 50          # steps per rollout per env
NUM_EVAL_STEPS = 50
GAMMA = 0.8             # official default for manipulation tasks
GAE_LAMBDA = 0.9        # official default
NUM_MINIBATCHES = 32
UPDATE_EPOCHS = 8       # official default: more gradient steps
CLIP_COEF = 0.2
ENT_COEF = 0.0          # official default: no entropy bonus
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.1         # official default: less restrictive than SB3's 0.05
REWARD_SCALE = 1.0
SEED = 1
EVAL_FREQ = 25          # evaluate every 25 iterations
FINITE_HORIZON_GAE = True  # THE critical feature SB3 lacks
ANNEAL_LR = False
PARTIAL_RESET = True    # auto-reset on termination (not truncation)
SAVE_FREQ = 25          # save checkpoint every 25 iterations

BATCH_SIZE = NUM_ENVS * NUM_STEPS          # 256 * 50 = 12,800
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES  # 400
NUM_ITERATIONS = TOTAL_TIMESTEPS // BATCH_SIZE

SIM_CONFIG = dict(
    gpu_memory_config=dict(
        max_rigid_patch_count=1048576,
        max_rigid_contact_count=2097152,
        collision_stack_size=67108864  # 64MB to prevent SAPIEN overflow
    )
)


# ===========================================================================
# Network — exact copy of official baseline Agent
# ===========================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ===========================================================================
# Main training loop — adapted from official ppo.py
# ===========================================================================
def main():
    run_name = f"{ENV_ID}__cleanrl_ppo__{SEED}__{int(time.time())}"

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Environment setup
    # -----------------------------------------------------------------------
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        sim_config=SIM_CONFIG,
    )

    import custom_reward
    custom_reward.patch_reward()
    envs = gym.make(ENV_ID, num_envs=NUM_ENVS, **env_kwargs)
    eval_envs = gym.make(ENV_ID, num_envs=NUM_EVAL_ENVS, reconfiguration_freq=1, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # ManiSkillVectorEnv handles partial resets and provides final_observation
    envs = ManiSkillVectorEnv(envs, NUM_ENVS, ignore_terminations=not PARTIAL_RESET, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, NUM_EVAL_ENVS, ignore_terminations=True, record_metrics=True)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # -----------------------------------------------------------------------
    # Agent and optimizer
    # -----------------------------------------------------------------------
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # -----------------------------------------------------------------------
    # Tensorboard
    # -----------------------------------------------------------------------
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{k}|{v}|" for k, v in {
            "env_id": ENV_ID, "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS, "num_steps": NUM_STEPS,
            "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
            "update_epochs": UPDATE_EPOCHS, "num_minibatches": NUM_MINIBATCHES,
            "learning_rate": LEARNING_RATE, "ent_coef": ENT_COEF,
            "finite_horizon_gae": FINITE_HORIZON_GAE,
            "partial_reset": PARTIAL_RESET,
        }.items()])
    ))

    # -----------------------------------------------------------------------
    # Storage tensors (all on GPU)
    # -----------------------------------------------------------------------
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=SEED)
    eval_obs, _ = eval_envs.reset(seed=SEED)
    next_done = torch.zeros(NUM_ENVS, device=device)

    print(f"####")
    print(f"num_iterations={NUM_ITERATIONS} num_envs={NUM_ENVS}")
    print(f"minibatch_size={MINIBATCH_SIZE} batch_size={BATCH_SIZE} update_epochs={UPDATE_EPOCHS}")
    print(f"finite_horizon_gae={FINITE_HORIZON_GAE} partial_reset={PARTIAL_RESET}")
    print(f"####")

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    for iteration in range(1, NUM_ITERATIONS + 1):
        final_values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        agent.eval()

        # -------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------
        if iteration % EVAL_FREQ == 1:
            print(f"\n=== Evaluation at iteration {iteration} (step {global_step}) ===")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(NUM_EVAL_STEPS):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"  {NUM_EVAL_STEPS * NUM_EVAL_ENVS} steps, {num_episodes} episodes completed")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                writer.add_scalar(f"eval/{k}", mean, global_step)
                print(f"  eval/{k} = {mean:.4f}")

        # Save checkpoint
        if iteration % SAVE_FREQ == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"  Checkpoint saved: {model_path}")

        # LR annealing
        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
            optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        # -------------------------------------------------------------------
        # Rollout collection
        # -------------------------------------------------------------------
        rollout_time = time.time()
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * REWARD_SCALE

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                # CRITICAL: Bootstrap value from final_observation for truncated episodes
                with torch.no_grad():
                    final_values[step, torch.arange(NUM_ENVS, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)

        rollout_time = time.time() - rollout_time

        # -------------------------------------------------------------------
        # GAE advantage estimation with proper truncation handling
        # -------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # Use final_values for bootstrapping at episode boundaries
                real_next_values = next_not_done * nextvalues + final_values[t]

                if FINITE_HORIZON_GAE:
                    # Finite horizon GAE — correct for fixed-length episodes
                    if t == NUM_STEPS - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + GAE_LAMBDA * lam_coef_sum
                    reward_term_sum = GAE_LAMBDA * GAMMA * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = GAE_LAMBDA * GAMMA * value_term_sum + GAMMA * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + GAMMA * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_not_done * lastgaelam

            returns = advantages + values

        # -------------------------------------------------------------------
        # PPO update
        # -------------------------------------------------------------------
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        update_time = time.time()

        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                if TARGET_KL is not None and approx_kl > TARGET_KL:
                    break

                mb_advantages = b_advantages[mb_inds]
                # Per-minibatch advantage normalization (replaces VecNormalize)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            if TARGET_KL is not None and approx_kl > TARGET_KL:
                break

        update_time = time.time() - update_time

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout_time", rollout_time, global_step)
        writer.add_scalar("time/update_time", update_time, global_step)

        if iteration % 10 == 0:
            print(f"Iter {iteration}/{NUM_ITERATIONS} | step {global_step} | SPS {sps} | "
                  f"v_loss {v_loss.item():.4f} | pg_loss {pg_loss.item():.4f} | "
                  f"explained_var {explained_var:.3f}")

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_path = f"runs/{run_name}/final_ckpt.pt"
    torch.save(agent.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    writer.close()
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main()
