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

  def cond_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      counter, _, mjx_d, _ = carry
      return counter < 5000 # mjx_d.time < 0.01

  def body_fun(carry : typing.Tuple[int, mjx.Model, mjx.Data, jax.Array]):
      counter, mjx_m, mjx_d, depths = carry
      mjx_d = mjx.step(mjx_m, mjx_d)
      depths = jax.lax.cond(
          counter % 100 == 0,
          lambda _: depths.at[counter // 100].set(vray(mjx_m, mjx_d, origins, directions)[0]),
          lambda _: depths,
          None
      )
      counter += 1
      return counter, mjx_m, mjx_d, depths

  depths = jax.numpy.zeros((50, CAMERASIZE**2), dtype=float)
  return jax.lax.while_loop(cond_fun, body_fun, (counter, mjx_m, mjx_d, depths))

# simulate
counter, mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
print(counter)
depth = jax.device_get(depth)
print(depth)

from PIL import Image
import numpy as np

# Convert depth to a numpy array
depth_array = np.array(depth)

# Determine the size of the square image
size = int(np.sqrt(depth_array.shape[1]))

# Find the minimum and maximum depth values across all frames
min_depth = depth_array.min()
max_depth = depth_array.max()

# Create a list to store the frames of the GIF
frames = []

for i in range(depth_array.shape[0]):
    # Reshape the depth array into a square image
    depth_image = depth_array[i].reshape((size, size))

    # Normalize the depth values to the range [0, 255] based on the global min and max
    depth_image = (depth_image - min_depth) / (max_depth - min_depth) * 255

    # Convert the depth image to uint8 data type
    depth_image = depth_image.astype(np.uint8)

    # Create a PIL Image from the depth array
    image = Image.fromarray(depth_image)

    # Append the image to the frames list
    frames.append(image)

# Save the frames as a GIF
frames[0].save("depth_animation.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)