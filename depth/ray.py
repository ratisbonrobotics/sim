import jax
import mujoco
from mujoco import mjx
import typing

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
CAMERASIZE = 8*8

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

    x, y = jax.numpy.meshgrid(jax.numpy.linspace(-1, 1, CAMERASIZE), jax.numpy.linspace(-1, 1, CAMERASIZE))
    origins = jax.numpy.stack([x.ravel(), y.ravel(), jax.numpy.ones(CAMERASIZE**2)], axis=1)
    directions = jax.numpy.column_stack((jax.numpy.zeros((CAMERASIZE**2,)),jax.numpy.zeros((CAMERASIZE**2,)),-jax.numpy.ones((CAMERASIZE**2,))))

    def cond_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        _, mjx_d, _ = carry
        return mjx_d.time < 0.01

    def body_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        mjx_m, mjx_d, depth = carry
        mjx_d = mjx.step(mjx_m, mjx_d)
        depth = vray(mjx_m, mjx_d, origins, directions)
        return mjx_m, mjx_d, depth

    return jax.lax.while_loop(cond_fun, body_fun, (mjx_m, mjx_d, (jax.numpy.zeros((CAMERASIZE**2), dtype=float), jax.numpy.zeros((CAMERASIZE**2), dtype=int))))

# simulate
mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
depth = jax.device_get(depth)
print(depth)

from PIL import Image
import numpy as np

# Convert depth[0] to a numpy array
depth_image = np.array(depth[0])

# Determine the size of the square image
size = int(np.sqrt(depth_image.shape[0]))

# Reshape the depth array into a square image
depth_image = depth_image.reshape((size, size))

# Normalize the depth values to the range [0, 255]
depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255

# Convert the depth image to uint8 data type
depth_image = depth_image.astype(np.uint8)

# Create a PIL Image from the depth array
image = Image.fromarray(depth_image)

# Save the image as a PNG file
image.save("depth_image.png")