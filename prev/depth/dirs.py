import jax

import jax
import jax.numpy as jnp

CAMERASIZE = 2*2

a = jnp.linspace(-1, 1, CAMERASIZE)
x, y = jnp.meshgrid(a, a)
b = jnp.stack([x.ravel(), y.ravel(), jnp.ones(CAMERASIZE*CAMERASIZE)], axis=1)
print(b)