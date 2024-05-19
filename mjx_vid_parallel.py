import mujoco
from mujoco import mjx
import jax
import mediapy as media
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Make model and convert to MJX model
model = mujoco.MjModel.from_xml_string(xml)
mjx_model = mjx.put_model(model)

duration = 3.8  # seconds
framerate = 60  # Hz
num_simulations = 8  # Number of parallel simulations

# Function to reset and randomize initial positions
def reset_and_randomize(rng):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    random_qpos = jax.random.uniform(rng, (model.nq,), minval=-5.1, maxval=5.1)
    data.qpos[:] = random_qpos
    return data

# Prepare RNG for each simulation
rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, num_simulations)

# Initialize and randomize data objects for each simulation
datas = [reset_and_randomize(rng) for rng in rngs]

# Jit the step function for MJX
jit_step = jax.jit(mjx.step)

# Initialize a renderer
renderer = mujoco.Renderer(model)

# Simulate and collect frames for each simulation
videos = []

for data in datas:
    frames = []
    mjx_data = mjx.put_data(model, data)
    while mjx_data.time < duration:
        mjx_data = jit_step(mjx_model, mjx_data)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(model, mjx_data)
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)
    videos.append(frames)

renderer.close()

# Save videos from frames for each simulation
for i, frames in enumerate(videos):
    media.write_video(f'simulation_mjx_{i + 1}.mp4', frames, fps=framerate)
