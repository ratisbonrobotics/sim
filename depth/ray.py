import jax
import mujoco
from mujoco import mjx
import mediapy as media

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
        mjx_d, _ = carry
        return mjx_d.time < 3.8

    def body_fun(carry):
        mjx_d, mjx_m = carry
        mjx_d = mjx.step(mjx_m, mjx_d)
        return mjx_d, mjx_m

    mjx_d, _ = jax.lax.while_loop(cond_fun, body_fun, (mjx_d, mjx_m))
    return mjx_d

# simulate
mjx_d = jax.jit(sim)(mjx_model, mjx_data)