import jax
import mujoco
from mujoco import mjx
import typing

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
    def cond_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        _, mjx_d, _ = carry
        return mjx_d.time < 0.01

    def body_fun(carry : typing.Tuple[mjx.Model, mjx.Data, typing.Tuple[jax.Array, jax.Array]]):
        mjx_m, mjx_d, depth = carry
        mjx_d = mjx.step(mjx_m, mjx_d)
        origin = jax.numpy.array([[0.0, 0.20, 1.0]], dtype=float)
        directions = jax.numpy.array([[0.0, 0.0, -1.0]], dtype=float)
        depth = vray(mjx_m, mjx_d, origin, directions)
        return mjx_m, mjx_d, depth

    return jax.lax.while_loop(cond_fun, body_fun, (mjx_m, mjx_d, (jax.numpy.zeros((1), dtype=float), jax.numpy.zeros((1), dtype=int))))

# simulate
mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
print(depth)