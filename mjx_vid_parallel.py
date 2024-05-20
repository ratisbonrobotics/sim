import mujoco
from mujoco import mjx
import jax

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

model = mujoco.MjModel.from_xml_string(xml)
mjx_model = mjx.put_model(model)

@jax.vmap
def batched_step(mjx_data):
  for _ in range(10):
    mjx.step(mjx_model, mjx_data)
  return mjx_data

# Function to reset and randomize initial positions
def reset_and_randomize(rng):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    random_qpos = jax.random.uniform(rng, (model.nq,), minval=-5.1, maxval=5.1)
    data.qpos[:] = random_qpos
    return mjx.put_data(model, data)

# Prepare RNG for each simulation
rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, 4)

# Initialize and randomize data objects for each simulation
datas = [reset_and_randomize(rng) for rng in rngs]

# Create batched data using jax.tree_map
batched_data = jax.tree.map(lambda *args: jax.numpy.stack(args), *datas)

batched_data = batched_step(batched_data)