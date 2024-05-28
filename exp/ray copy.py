import jax
import mujoco
from mujoco import mjx
import typing
import mediapy as mp
import numpy as np

xml = """
<mujoco>
<asset>
<material name="shiny_red" rgba="1 0 0 1" specular="1"/>
<material name="matte_green" rgba="0 1 0 1" shininess="0.1"/>
</asset>
<worldbody>
<light name="top" pos="0 0 1"/>
<body name="box_and_sphere" euler="0 0 -30">
<joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
<geom name="red_box" type="box" size=".2 .2 .2" material="shiny_red"/>
<geom name="green_sphere" pos=".2 .2 .2" size=".1" material="matte_green"/>
</body>
</worldbody>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Move to accelerator
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# Define sim function
def sim(mjx_m: mjx.Model, mjx_d: mjx.Data, camera_origin: jax.Array, camera_directions: jax.Array):
  def cond_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
    _, mjx_d, _ = carry
    return mjx_d.time < 3.8

  def body_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
    mjx_m, mjx_d, depth = carry
    mjx_d = mjx.step(mjx_m, mjx_d)
    # Calculate ray destinations dynamically
    ray_destinations = camera_origin + camera_directions
    # vmap the ray function
    depth = jax.vmap(lambda x: mjx.ray(mjx_m, mjx_d, camera_origin, x))(ray_destinations)
    return mjx_m, mjx_d, depth

  return jax.lax.while_loop(cond_fun, body_fun, (mjx_m, mjx_d, (jax.numpy.zeros((len(camera_directions),), dtype=float), jax.numpy.zeros((len(camera_directions),), dtype=int))))

# Set up camera parameters
camera_origin = jax.numpy.array([0.0, 0.0, 0.0], dtype=float) # Camera position
resolution = 512
# Generate a grid of ray directions for the depth map
x, y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
camera_directions = jax.numpy.array(np.stack((x.flatten(), y.flatten(), np.ones(resolution * resolution)), axis=-1), dtype=float)

# Simulate
mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data, camera_origin, camera_directions)

# Reshape depth information into a 512x512 depth map
depth_map = np.array(depth[0]).reshape(resolution, resolution)

# Normalize depth values to be in the range [0, 255]
depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255

# Convert to uint8 for saving as a PNG
depth_map = depth_map.astype(np.uint8)

# Save the depth map
mp.write_image("depth_map.png", depth_map)