import mujoco
from mujoco import mjx
import imageio
import jax

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

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
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
with imageio.get_writer('simulation_mjx.mp4', fps=60) as video:
    while mjx_data.time < duration:
        mjx_data = jit_step(mjx_model, mjx_data)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(model, mjx_data)
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)
            video.append_data(pixels)

renderer.close()