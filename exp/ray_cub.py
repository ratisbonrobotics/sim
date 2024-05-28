import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

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

width, height = 128, 128
fovy = jp.deg2rad(58)
f = 0.1

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

h_ip = jp.tan(fovy / 2) * 2 * f
w_ip = h_ip * (width / height)
delta = w_ip / (2 * width)
xs = jp.linspace(-w_ip / 2 + delta, w_ip / 2 - delta, width)
ys = jp.flip(jp.linspace(-h_ip / 2 + delta, h_ip / 2 - delta, height))
xs, ys = jp.tile(xs, height), jp.repeat(ys, width)

cam_x, cam_y = xs, ys
cam_vec = jax.vmap(lambda x, y: data.cam_xmat @ jp.array([x, y, -f]).reshape(3, 1))(cam_x, cam_y)
cam_vec = jax.vmap(lambda x: x.flatten() / jp.linalg.norm(x))(cam_vec)
cam_pos = data.cam_xpos

def render():
    def fn(_, vec):
        dist, _ = mjx.ray(model, data, cam_pos, vec)
        return None, dist
    _, dist = jax.lax.scan(fn, None, cam_vec)
    return dist

depth = render().reshape(height, width)

from PIL import Image
Image.fromarray(depth * 100, "F").convert("RGB").resize((256, 256))