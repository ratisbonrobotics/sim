import jax
import jax.numpy as jp
from mujoco import mjx
from brax.io import mjcf

def _load_sys(path: str) -> mjcf.RootElement:
  xml = path.read_text()
  model = mjx.load_model_from_xml(xml)
  data = mjx.MjData(model)
  return model, data

model, data = _load_sys('./mujoco_menagerie/aloha/mjx_single_cube.xml')

width, height = 128, 128
fovy = jp.deg2rad(58)
f = 0.1
cam_id = 1

h_ip = jp.tan(fovy/2) * 2 * f
w_ip = h_ip * (width / height)
delta = w_ip / (2 * width)
xs = jp.linspace(-w_ip/2 + delta, w_ip/2 - delta, width)
ys = jp.flip(jp.linspace(-h_ip/2 + delta, h_ip/2 - delta, height))
xs, ys = jp.tile(xs, height), jp.repeat(ys, width)

cam_x, cam_y = xs, ys
cam_vec = jax.vmap(lambda x, y: data.cam_xmat[cam_id] @ jp.array([x, y, -f]))(cam_x, cam_y)
cam_vec = jax.vmap(lambda x: x / jp.linalg.norm(x))(cam_vec)
cam_pos = data.cam_xpos[cam_id]

@jax.jit
def render():
  geomgroup = [True, True, True, False, False, False]
  def fn(_, vec):
    dist, _ = mjx.ray(model, data, cam_pos, vec, geomgroup)
    return None, dist
  _, dist = jax.lax.scan(fn, None, cam_vec)
  return dist

depth = render().reshape(height, width)

from PIL import Image
Image.fromarray(depth * 100, "F").convert("RGB").resize((256, 256))