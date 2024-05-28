import jax
import mujoco
from mujoco import mjx

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
def sim(mjx_m: mjx.Model, mjx_d: mjx.Data):
    def cond_fun(carry):
        _, mjx_d, _ = carry
        return mjx_d.time < 3.8

    def body_fun(carry):
        mjx_m, mjx_d, depth = carry
        mjx_d = mjx.step(mjx_m, mjx_d)
        depth = mjx.ray(mjx_m, mjx_d, jax.lax.full((3,), 0.0, dtype=jax.numpy.float32), jax.lax.full((3,), 1.0, dtype=jax.numpy.float32))
        return mjx_m, mjx_d, depth

    depth = (jax.lax.full((), 0.0, dtype=jax.numpy.float32), jax.lax.full((), 0, dtype=jax.numpy.int32))
    mjx_m, mjx_d, depth = jax.lax.while_loop(cond_fun, body_fun, (mjx_m, mjx_d, depth))
    return mjx_m, mjx_d, depth

# simulate
mjx_m, mjx_d, depth = jax.jit(sim)(mjx_model, mjx_data)
print(depth)