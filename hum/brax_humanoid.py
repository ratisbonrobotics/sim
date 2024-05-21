from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from matplotlib import pyplot as plt
from etils import epath
import mediapy as media
import mujoco
from mujoco import mjx


class Humanoid(PipelineEnv):

    def __init__(self, **kwargs):
        mj_model = mujoco.MjModel.from_xml_path((epath.Path(epath.resource_path("mujoco")) / ("mjx/test_data/humanoid") / "humanoid.xml").as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        reset_noise_scale = 1e-2
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -reset_noise_scale, reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {"forward_reward": zero, "reward_linvel": zero, "reward_quadctrl": zero, "reward_alive": zero, "x_position": zero, "y_position": zero, "distance_from_origin": zero, "x_velocity": zero, "y_velocity": zero,}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = 1.25 * velocity[0]

        min_z, max_z = 1.0, 2.0
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0 if is_healthy else 0.0

        ctrl_cost = 0.1 * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy

        state.metrics.update(forward_reward=forward_reward, reward_linvel=forward_reward, reward_quadctrl=-ctrl_cost, reward_alive=healthy_reward, x_position=com_after[0], y_position=com_after[1], distance_from_origin=jp.linalg.norm(com_after), x_velocity=velocity[0], y_velocity=velocity[1])
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        exclude_current_positions_from_observation = True
        position = data.qpos
        if exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )


train_fn = functools.partial(
    ppo.train,
    num_timesteps=30_000_000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=16,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=1024,
    batch_size=512,
    seed=0,
)

envs.register_environment("humanoid_mjx", Humanoid)
make_inference_fn, params, _ = train_fn(
    environment=envs.get_environment("humanoid_mjx")
)

model.save_params("/home/markusheimerl/mjx_brax_policy2", params)
