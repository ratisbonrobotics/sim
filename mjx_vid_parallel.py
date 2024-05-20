import mujoco
from mujoco import mjx
import jax
from PIL import Image

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

mj_model = mujoco.MjModel.from_xml_string(xml)
mjx_model = mjx.put_model(mj_model)

def batched_step(mjx_data):
  for _ in range(10):
    mjx_data = mjx.step(mjx_model, mjx_data)
  return mjx_data

# Initialize and randomize data objects for each simulation
n_sim = 8
datas = [mjx.make_data(mjx_model) for _ in range(n_sim)]
models = [mjx_model for _ in range(n_sim)]

# Create batched data using jax.tree_map
batched_data = jax.tree.map(lambda *args: jax.numpy.stack(args), *datas)
batched_data = jax.vmap(batched_step)(batched_data)

# Render result of simulation
for i in range(n_sim):
  single_data = jax.tree.map(lambda x: x[i], batched_data)

  renderer = mujoco.Renderer(mj_model)
  renderer.update_scene(mjx.get_data(mj_model, single_data))
  pixels = renderer.render()
  renderer.close()

  # Save pixels as PNG
  image = Image.fromarray(pixels)
  image.save(f"imgs/rendered_image_{i}.png")