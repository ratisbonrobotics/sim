import mujoco
from mujoco import mjx
import jax
import mediapy as media

# Make model and data
model = mujoco.MjModel.from_xml_path("car.xml")
data = mujoco.MjData(model)

mjx_model = mjx.put_model(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

jit_step = jax.jit(mjx.step)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
mjx_data = mjx.put_data(model, data)
while mjx_data.time < duration:
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(model, mjx_data)
        renderer.update_scene(mj_data)
        pixels = renderer.render()
        frames.append(pixels)

renderer.close()

# Save video from frames
media.write_video('car_mjx.mp4', frames, fps=framerate)