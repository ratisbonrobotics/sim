import mujoco
import imageio

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <camera name="cam" pos="0 0 1" xyaxes="0 1 0 -1 0 0"/>
    <body name="box_and_sphere">
      <joint name="swing" type="hinge" axis="1 -1 0" pos=".2 .0 .2"/>
      <geom name="red_box" type="box" pos="0 0 0" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .0 .2" size=".01" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)

duration = 2.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
with imageio.get_writer('simulation_mujoco.mp4', fps=60) as video:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, "cam")
            pixels = renderer.render()
            frames.append(pixels)
            video.append_data(pixels)

renderer.close()