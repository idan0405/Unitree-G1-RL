# G1 Manipulation — Place an Apple in a Bowl (ManiSkill 3 + PPO)

Train the Unitree G1 humanoid to **reach for an apple, grasp it, carry it over a
bowl, and release it** so it comes to rest inside, using
[ManiSkill 3](https://maniskill.readthedocs.io/)'s
`UnitreeG1PlaceAppleInBowl-v1` task. Training uses a custom GPU-resident PPO
implementation (CleanRL-style) with thousands of parallel environments, plus a
re-shaped dense reward that makes this long-horizon, high-DOF task learnable.

> **GPU required.** The trainer uses ManiSkill's `physx_cuda` backend and keeps
> all rollouts and updates on the GPU. An NVIDIA GPU with CUDA is needed.

## Training progression

![G1 apple-in-bowl training progression](media/training_progression.gif)

Ten checkpoints played in sequence, weighted toward the early phase where the
behaviour actually changes. The policy learns the task in stages: random
flailing (0M) → reaching the apple (~5M) → grasping it (~10M) → and finally
lifting it into the bowl (~26M). Past ~31M steps it places the apple reliably,
so only one late checkpoint (99M) is shown. Full-resolution clip:
[training_progression.mp4](media/training_progression.mp4).

## The task

A 25-DOF G1 (mobile base + arm + dexterous hand) must complete a multi-phase
manipulation: **reach → grasp → transport → release → settle in bowl**. Sparse
success alone is far too hard to discover, so the reward is densely shaped.

### Reward shaping (`custom_reward.py`)

`patch_reward()` monkey-patches the ManiSkill task's `compute_dense_reward` and
`evaluate` with staged shaping:

1. **Reaching** — `1 − tanh(5 · ‖tcp − apple‖)` pulls the hand to the apple.
2. **Grasp bonus** — `+1` once the apple is grasped.
3. **Placing** — while grasped, reward shrinking distance from the apple to a
   point ~15 cm above the bowl.
4. **Release / settle** — once above the bowl (or the apple is already over/in
   it), switch to a higher reward band that rewards the apple approaching the
   bowl, keeping the hand near the bowl, and releasing the grasp. This
   deliberately removes the "reward cliff" between *carrying* and *dropping*.
5. **Success** — large terminal bonus (`+50`).

The patched `evaluate()` also widens the placement tolerance (0.05 → 0.08 m) so
an apple resting slightly off-center in the bowl is not falsely scored as a
failure (the Euclidean check includes the apple's resting height).

## Setup

```bash
cd manipulation
pip install -r requirements.txt
# ManiSkill 3 install + GPU/driver setup: https://maniskill.readthedocs.io/
```

Run all scripts from this folder.

## Usage

```bash
# Quick end-to-end smoke run (~25k steps, small) — verify the whole pipeline works
python smoke_train.py

# Full training run (100M steps, 512 parallel envs)
python train.py

# Render the latest checkpoint and print a running success rate
python evaluate.py
```

Checkpoints and TensorBoard logs are written under
`runs/UnitreeG1PlaceAppleInBowl-v1__cleanrl_ppo__<seed>__<timestamp>/`.
`evaluate.py` automatically loads the most recent run's `final_ckpt.pt` (or the
latest `ckpt_*.pt`).

```bash
tensorboard --logdir runs
```

## Why a custom PPO (and not just Stable-Baselines3)?

The trainer in `train.py` is adapted from ManiSkill's official PPO baseline and
keeps the features that matter for GPU-parallel, fixed-horizon manipulation:

- **Stays on the GPU** — no per-step CPU↔GPU transfers across thousands of envs.
- **`ManiSkillVectorEnv`** with partial resets and `final_observation`, so values
  are correctly **bootstrapped on truncation** (not just termination).
- **Finite-horizon GAE** — correct advantage estimation for fixed-length episodes.
- **Per-minibatch advantage normalization** instead of `VecNormalize`.
- **Linear LR annealing** for stable late-stage convergence.

The original, unmodified baseline is kept in
[`reference/official_ppo.py`](reference/official_ppo.py) (driven by a `tyro` CLI)
for side-by-side comparison.

## Key hyperparameters (`train.py`)

| Param | Value | Notes |
|-------|-------|-------|
| parallel envs | 512 | reduce if you hit GPU OOM |
| total timesteps | 100M | hard multi-phase task |
| rollout steps | 100 | matches `max_episode_steps` |
| minibatches / epochs | 4 / 4 | few, large (~12.8k) minibatches |
| learning rate | 3e-4 | linearly annealed to 0 |
| gamma / gae_lambda | 0.8 / 0.9 | ManiSkill manipulation defaults |
| ent_coef | 0.01 | escape the reaching-only local optimum |
| target_kl | 0.1 | guard against destructive updates in 25-DOF |
| finite-horizon GAE | on | |

`smoke_train.py` mirrors `train.py` with much smaller numbers for a fast
sanity check.

## Files

| Path | Purpose |
|------|---------|
| `train.py` | full GPU PPO training run |
| `smoke_train.py` | small, fast end-to-end smoke run |
| `custom_reward.py` | staged dense-reward + evaluate patches for the place task |
| `evaluate.py` | load the latest checkpoint and render with a success counter |
| `reference/official_ppo.py` | unmodified ManiSkill PPO baseline (reference) |
| `tools/check_control_modes.py` | print the env's action/obs spaces and control modes |
| `tools/arxiv_search.py` | quick arXiv query helper used during research |
| `tests/` | exploratory sanity scripts (reward geometry, dense reward, pickling, SB3 vector wrapper) |
