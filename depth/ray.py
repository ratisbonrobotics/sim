import jax
import mujoco
from mujoco import mjx
import typing
from PIL import Image

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

# vmap mjx.ray
vray = jax.vmap(mjx.ray, (None, None, None, 0), (0, 0))

# Define camera parameters
fov = 60.0  # Field of view in degrees
resolution = (64, 64)  # Camera resolution (width, height)

# Generate camera directions
def generate_camera_directions(fov, resolution):
    fov_rad = jax.numpy.deg2rad(fov)
    aspect_ratio = resolution[0] / resolution[1]
    x = jax.numpy.linspace(-0.5, 0.5, resolution[0]) * jax.numpy.tan(fov_rad / 2) * aspect_ratio
    y = jax.numpy.linspace(-0.5, 0.5, resolution[1]) * jax.numpy.tan(fov_rad / 2)
    xx, yy = jax.numpy.meshgrid(x, y)
    directions = jax.numpy.stack((xx, yy, -jax.numpy.ones_like(xx)), axis=-1)
    directions /= jax.numpy.linalg.norm(directions, axis=-1, keepdims=True)
    return directions

# Define sim function
def sim(mjx_m: mjx.Model, mjx_d: mjx.Data):
    def cond_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        _, mjx_d, _ = carry
        return mjx_d.time < 3.8

    def body_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        mjx_m, mjx_d, depth = carry
        mjx_d = mjx.step(mjx_m, mjx_d)
        origin = jax.numpy.array([0.0, 0.0, 0.0], dtype=float)
        directions = generate_camera_directions(fov, resolution).reshape(-1,3)
        depth = vray(mjx_m, mjx_d, origin, directions)
        return mjx_m, mjx_d, depth

    return jax.lax.while_loop(cond_fun, body_fun, (mjx_m, mjx_d, (jax.numpy.zeros((resolution[0] * resolution[1]), dtype=float), jax.numpy.zeros((resolution[0] * resolution[1]), dtype=int))))

# simulate
mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)

# Reshape depth values to match the resolution
depth_values = jax.device_get(depth[0]).reshape(resolution)

# Normalize depth values to [0, 255]
depth_normalized = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
depth_normalized = (depth_normalized * 255).astype(jax.numpy.uint8)

# Create a PIL Image object and save as PNG
depth_image = Image.fromarray(depth_normalized)
depth_image.save("depth.png")