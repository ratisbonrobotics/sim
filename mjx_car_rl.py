import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import mediapy as media
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs.base import Env, PipelineEnv, State

class CarEnv(PipelineEnv):
    def __init__(self, **kwargs):
        model = mujoco.MjModel.from_xml_path("car.xml")
        sys = mjx.put_model(model)
        super().__init__(sys, **kwargs)

    def reset(self, rng: jnp.ndarray) -> State:
        data = self.pipeline_init(self.sys.qpos0, self.sys.qvel0)
        obs = self._get_obs(data)
        reward, done, zero = jnp.zeros(3)
        metrics = {}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        data = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(data)
        reward = self._get_reward(data)
        done = jnp.where(data.time > 3.8, 1.0, 0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jnp.ndarray:
        # Get camera image as observation
        camera_id = mujoco.mj_name2id(self.sys.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
        image = mjx.render_camera(self.sys.model, data, camera_id, 84, 84)
        return image.flatten()

    def _get_reward(self, data: mjx.Data) -> jnp.ndarray:
        # Calculate reward based on distance to the red cube
        car_pos = data.qpos[:3]
        cube_pos = data.qpos[7:10]
        distance = jnp.linalg.norm(car_pos - cube_pos)
        reward = jnp.where(distance < 0.1, 1.0, 0.0)
        return reward

# Instantiate the environment
env = CarEnv()

# Define the network architecture
network = ppo_networks.make_model(
    observation_size=env.observation_size,
    action_size=env.action_size,
    hidden_layer_sizes=(64, 64),
)

# Train the agent
make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=100_000,
    num_evals=5,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=16,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=64,
    batch_size=256,
    seed=0,
    network=network,
)

# Evaluate the trained agent
duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video
frames = []
state = env.reset(jax.random.PRNGKey(0))
while state.pipeline_state.time < duration:
    action = make_inference_fn(params, state.obs)
    state = env.step(state, action)
    if len(frames) < state.pipeline_state.time * framerate:
        mj_data = mjx.get_data(env.sys.model, state.pipeline_state)
        renderer = mujoco.Renderer(env.sys.model)
        renderer.update_scene(mj_data)
        pixels = renderer.render()
        frames.append(pixels)
        renderer.close()

# Save video from frames
media.write_video('car_rl_mjx.mp4', frames, fps=framerate)