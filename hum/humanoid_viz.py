import jax
import mujoco
import mediapy
import functools
from brax import envs
from mujoco import mjx
from etils import epath
from jax import numpy as jp
from brax.io import mjcf, model
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo

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
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -1e-2, 1e-2 # reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero, "reward_linvel": zero, "reward_quadctrl": zero, "reward_alive": zero,
            "x_position": zero, "y_position": zero,"distance_from_origin": zero,"x_velocity": zero,"y_velocity": zero
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = 1.25 * velocity[0]

        min_z, max_z = (1.0, 2.0) # healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        healthy_reward = 5.0
        ctrl_cost = (0.1 * jp.sum(jp.square(action)))

        state.metrics.update(
            forward_reward=forward_reward, reward_linvel=forward_reward, reward_quadctrl=-ctrl_cost, 
            reward_alive=healthy_reward, x_position=com_after[0], y_position=com_after[1], 
            distance_from_origin=jp.linalg.norm(com_after), x_velocity=velocity[0], y_velocity=velocity[1]
        )
        return state.replace(
            pipeline_state=data, obs=self._get_obs(data, action), 
            reward=(forward_reward + healthy_reward - ctrl_cost), done=(1.0 - is_healthy)
        )

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        return jp.concatenate([data.qpos[2:], data.qvel, data.cinert[1:].ravel(), data.cvel[1:].ravel(), data.qfrc_actuator])


train_fn = functools.partial(
    ppo.train, num_timesteps=1, num_evals=1, reward_scaling=0.1, episode_length=1, normalize_observations=True,
    action_repeat=1, unroll_length=10, num_minibatches=1, num_updates_per_batch=1, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3,
    num_envs=1, batch_size=1, seed=0
)

print("Registering enviornment...")
envs.register_environment("humanoid_mjx", Humanoid)
env = envs.get_environment("humanoid_mjx")
print("Running pseudo training...")
make_inference_fn, params, _ = train_fn(environment=env)
print("Loading...")
params = model.load_params("/home/markusheimerl/mjx_brax_policy_v2")
print("Compiling...")
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# initialize the state
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 100
print("Rendering...")

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)
    if state.done:
        break

mediapy.write_video("humanoid_viz.mp4", env.render(rollout, camera="side"), fps=60)