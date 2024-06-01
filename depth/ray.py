import jax
import mujoco
from mujoco import mjx
import typing
import mediapy

#jax.config.update("jax_compilation_cache_dir", "/home/markusheimerl/sim/cache")

xml = """
<mujoco>
  <asset>
    <material name="shiny_red" rgba="1 0 0 1" specular="1"/>
    <material name="matte_green" rgba="0 1 0 1" shininess="0.1"/>
  </asset>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere">
      <joint name="swing" type="hinge" axis="1 -1 0" pos=".2 .0 .2"/>
      <geom name="red_box" type="box" pos="0 0 0" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .0 .2" size=".01" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
CAMERASIZE = 128

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Move to accelerator
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# vmap mjx.ray
vray = jax.vmap(mjx.ray, (None, None, 0, 0), (0, 0))

# Define sim function
def sim(mjx_m: mjx.Model, mjx_d: mjx.Data):
  x, y = jax.numpy.meshgrid(jax.numpy.linspace(-0.5, 0.5, CAMERASIZE), jax.numpy.linspace(-0.5, 0.5, CAMERASIZE))
  origins = jax.numpy.stack([x.ravel(), y.ravel(), jax.numpy.ones(CAMERASIZE**2)], axis=1)
  directions = jax.numpy.column_stack((jax.numpy.zeros((CAMERASIZE**2,)),jax.numpy.zeros((CAMERASIZE**2,)),-jax.numpy.ones((CAMERASIZE**2,))))
  counter = 0
  end_time = 1.0

  def cond_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      _, _, mjx_d, _ = carry
      return mjx_d.time < end_time

  def body_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      counter, mjx_m, mjx_d, depths = carry
      mjx_d = mjx.step(mjx_m, mjx_d) # one step steps for 2 ms
      depths = depths.at[counter].set(vray(mjx_m, mjx_d, origins, directions)[0])
      counter += 1
      return counter, mjx_m, mjx_d, depths

  depths = jax.numpy.zeros((int(end_time / 0.002), CAMERASIZE**2), dtype=float)
  return jax.lax.while_loop(cond_fun, body_fun, (counter, mjx_m, mjx_d, depths))

# simulate
counter, mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
depth = jax.device_get(depth)

# save the frames as MP4
min_depth = depth.min()
max_depth = depth.max()
frames = []

for i in range(depth.shape[0]):
    depth_image = depth[i].reshape((int(jax.numpy.sqrt(depth.shape[1])), int(jax.numpy.sqrt(depth.shape[1]))))
    depth_image = (depth_image - min_depth) / (max_depth - min_depth) * 255
    depth_image = depth_image.astype(jax.numpy.uint8)
    frames.append(depth_image)

mediapy.write_video("depth/vid1.mp4", frames, fps=1.0 / 0.002)