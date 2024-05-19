import mujoco
from mujoco import mjx, Renderer
import numpy as np
import imageio

XML = """
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)

velocities = np.linspace(0.0, 1.0, 2)
renderer = Renderer(model)

with imageio.get_writer('simulation.mp4', fps=30) as video:
    for vel in velocities:
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[0].set(vel))
        mjx_data = mjx.step(mjx_model, mjx_data)
        mj_data = mjx.get_data(model, mjx_data)
        renderer.render(mj_data)
        frame = renderer.read_pixels()[0]
        video.append_data(frame)

print("Video saved successfully.")
