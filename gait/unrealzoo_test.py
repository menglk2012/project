import os
import sys
from collections import deque

import gym
import numpy as np
import torch
import torch.nn.functional as F
from dm_env import specs, StepType

from aesthetics_model import AestheticsModel
from habitat_test import ExtendedTimeStep, Runner, AsyncRunners, make_async_runners


_UNREALZOO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unrealzoo-gym"))
if _UNREALZOO_PATH not in sys.path:
    sys.path.append(_UNREALZOO_PATH)

import gym_unrealcv  # noqa: E402,F401
from gym_unrealcv.envs.wrappers import configUE  # noqa: E402


class UnrealZooGymWrapper:
    """Gym-style wrapper that mimics HabitatSimGymWrapper's reset/step interface."""

    def __init__(self, cfg):
        self.scene_name = cfg.scene_name
        self.pose_dim = cfg.pose_dim
        self.max_timestep = cfg.max_timestep
        self.t = 0

        env_id = cfg.unrealzoo_env_id or f"UnrealAgent-{self.scene_name}-ContinuousColor-v0"
        self.env = gym.make(env_id)
        self.env = configUE.ConfigUEWrapper(
            self.env,
            offscreen=cfg.unrealzoo_offscreen,
            resolution=(cfg.state_dim[1], cfg.state_dim[2]),
            gpu_id=cfg.unrealzoo_gpu_id,
        )

    def _to_hwc_image(self, obs):
        """Normalize UnrealZoo observation into uint8 HWC image."""
        arr = np.asarray(obs)
        # Remove leading singleton dims and pick first sample if a batch/agent axis exists.
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected observation shape: {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr

    def _pack_obs(self, obs):
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        return self._to_hwc_image(obs)

    def _extract_pose(self, info):
        # UnrealZoo usually returns [x, y, z, roll, yaw, pitch]; map to 5D [x, y, z, yaw, pitch].
        pose = np.zeros((self.pose_dim,), dtype=np.float32)
        if info is None or "Pose" not in info or len(info["Pose"]) == 0:
            return pose
        raw_pose = np.asarray(info["Pose"][0], dtype=np.float32)
        pose[:3] = raw_pose[:3]
        if self.pose_dim > 3 and raw_pose.shape[0] >= 6:
            pose[3] = raw_pose[4]  # yaw
            pose[4] = raw_pose[5]  # pitch
        return pose

    def reset(self, eval_i=None, uniform=False, to_pose=None, apply_filter=False, to_quat=None):
        del eval_i, uniform, to_pose, apply_filter, to_quat
        self.t = 0
        obs = self.env.reset()
        img = self._pack_obs(obs)
        pose = np.zeros((self.pose_dim,), dtype=np.float32)
        return img, pose, False, img

    def step(self, action, apply_filter=False):
        del apply_filter
        self.t += 1
        if isinstance(self.env.action_space, list):
            action = [np.asarray(action, dtype=np.float32)]
        obs, reward, done, info = self.env.step(action)
        img = self._pack_obs(obs)
        pose = self._extract_pose(info)
        done = bool(done) or self.t >= self.max_timestep
        return img, pose, done, img

    def close(self):
        self.env.close()


class UnrealZooDMCWrapper:
    """dm_env-like wrapper aligned with HabitatSimDMCWrapper outputs."""

    def __init__(self, cfg, scene_index=0, aesthetics_model=None):
        del scene_index
        self.cfg = cfg
        self.env = UnrealZooGymWrapper(cfg)
        self.discount = cfg.discount
        self.pose_dim = cfg.pose_dim
        self.max_timestep = cfg.max_timestep
        self.aesthetics_model = aesthetics_model if aesthetics_model is not None else AestheticsModel(
            negative_reward=None, device=cfg.device
        )

        self.observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, "observation")
        self.pose_spec = specs.Array((cfg.pose_dim,), np.float32, "pose")
        self.t_spec = specs.Array((1,), np.float32, "t")
        self.action_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "action")

        self.use_context = cfg.use_context
        self.hist_len = cfg.agent.context_history_length
        if self.use_context:
            self.history_obs = deque(maxlen=self.hist_len)
            self.history_others = deque(maxlen=self.hist_len)

    def _make_aes_tensor(self, aes_obs):
        aes_obs = self.env._to_hwc_image(aes_obs)
        aes_tensor = torch.from_numpy(aes_obs).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        target_h = int(getattr(self.cfg, "aes_obs_height", aes_tensor.shape[-2]))
        target_w = int(getattr(self.cfg, "aes_obs_width", aes_tensor.shape[-1]))
        if aes_tensor.shape[-2] != target_h or aes_tensor.shape[-1] != target_w:
            aes_tensor = F.interpolate(aes_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return aes_tensor

    def reset(self, eval_i=None, uniformSampling=False, to_pose=None):
        img, pos, done, aes_obs = self.env.reset(eval_i=eval_i, uniform=uniformSampling, to_pose=to_pose)
        del done
        img = np.moveaxis(img, -1, 0)
        t = np.array([0.0], dtype=np.float32)
        action = np.zeros(self.action_spec.shape, dtype=np.float32)
        reward = 0.0
        aes_tensor = self._make_aes_tensor(aes_obs)
        ts = ExtendedTimeStep(StepType.FIRST, reward, self.discount, img, pos.astype(np.float32), action, t=t, aes_obs=aes_tensor)

        history = None
        if self.use_context:
            self.history_obs.clear()
            self.history_others.clear()
            self.history_obs.append(img)
            self.history_others.append(np.concatenate([action, np.full((1,), reward, dtype=np.float32), pos.astype(np.float32)]))
            history = [np.array(self.history_others), np.array(self.history_obs)]
        return ts, history

    def step(self, action):
        img, pos, done, aes_obs = self.env.step(action)
        img = np.moveaxis(img, -1, 0)
        t = np.array([self.env.t / self.max_timestep], dtype=np.float32)
        step_type = StepType.LAST if done else StepType.MID
        aes_tensor = self._make_aes_tensor(aes_obs)
        _, reward = self.aesthetics_model(aes_tensor, np.zeros((3,), dtype=np.float32), done)
        reward = float(reward)
        discount = 0.0 if done else self.discount
        ts = ExtendedTimeStep(step_type, reward, discount, img, pos.astype(np.float32), action.astype(np.float32), t=t, aes_obs=aes_tensor)

        history = None
        if self.use_context:
            self.history_obs.append(img)
            self.history_others.append(np.concatenate([action.astype(np.float32), np.full((1,), reward, dtype=np.float32), pos.astype(np.float32)]))
            history = [np.array(self.history_others), np.array(self.history_obs)]
        return ts, history

    def close(self):
        self.env.close()


class MultiSceneWrapper:
    """Lightweight MultiScene wrapper to keep interfaces consistent."""

    def __init__(self, cfg):
        self.cfg = cfg
        aesthetics_model = AestheticsModel(negative_reward=None, device=cfg.device)
        self.envs = [UnrealZooDMCWrapper(cfg, i, aesthetics_model=aesthetics_model) for i in range(cfg.num_scenes)]
        self.observation_spec = self.envs[0].observation_spec
        self.pose_spec = self.envs[0].pose_spec
        self.t_spec = self.envs[0].t_spec
        self.action_spec = self.envs[0].action_spec

    def reset(self, eval_i=None, to_poses=None):
        del to_poses
        time_steps, histories = [], []
        for env in self.envs:
            ts, hist = env.reset(eval_i=eval_i)
            time_steps.append(ts)
            histories.append(hist)
        return time_steps, histories

    def step(self, actions):
        time_steps, histories = [], []
        for env, action in zip(self.envs, actions):
            ts, hist = env.step(action)
            time_steps.append(ts)
            histories.append(hist)
        return time_steps, histories

    def close(self):
        for env in self.envs:
            env.close()


class AestheticTourDMCWrapper:
    """Compatibility wrapper exposing the same fields as habitat AestheticTourDMCWrapper."""

    def __init__(self, cfg, data_specs=None):
        del data_specs
        self.cfg = cfg
        self.pose_dim = cfg.pose_dim
        self.num_scenes = cfg.num_scenes
        self.observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, "observation")
        self.pose_spec = specs.Array((cfg.pose_dim,), np.float32, "pose")
        self.t_spec = specs.Array((1,), np.float32, "t")
        self.action_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "action")

        d_pose_shape = cfg.pose_dim
        if cfg.distance_obs:
            d_pose_shape += 1
        if cfg.rand_diversity_radius:
            d_pose_shape += 1
        self.excluding_seq_spec = specs.BoundedArray((cfg.num_excluding_sequences, d_pose_shape), np.float32, -1.0, 1.0, "excluding_seq")

        self.step_size_dim = 3 if cfg.position_only_smoothness else cfg.pose_dim
        if cfg.smoothness_window > 0:
            self.avg_step_size_spec = specs.BoundedArray((cfg.smoothness_window, self.step_size_dim), np.float32, -1.0, 1.0, "avg_step_size")
        else:
            self.avg_step_size_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "avg_step_size")

        self.env = MultiSceneWrapper(cfg)

    def _wrap_timestep(self, ts):
        exc_seq = np.zeros(self.excluding_seq_spec.shape, dtype=np.float32)
        avg_step_size = np.zeros(self.avg_step_size_spec.shape, dtype=np.float32)
        return ExtendedTimeStep(
            ts.step_type,
            ts.reward,
            ts.discount,
            ts.observation,
            ts.pose,
            ts.action,
            ts.t,
            exc_seq,
            ts.aes_obs,
            1.0,
            avg_step_size,
            1.0,
        )

    def reset(self, eval_i=None, to_poses=None, curr_excluding_seqs=None, curr_sequence_i=None, curr_step_sizes=None, curr_diversity_radius=None):
        del to_poses, curr_excluding_seqs, curr_sequence_i, curr_step_sizes, curr_diversity_radius
        time_steps, histories = self.env.reset(eval_i=eval_i)
        return [self._wrap_timestep(ts) for ts in time_steps], histories

    def step(self, actions):
        time_steps, histories = self.env.step(actions)
        return [self._wrap_timestep(ts) for ts in time_steps], histories

    def close(self):
        self.env.close()
