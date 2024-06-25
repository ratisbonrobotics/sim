import jax


a = jax.numpy.linspace(-1, 1, 8*8)
b = jax.numpy.column_stack((a,a,jax.numpy.zeros((8*8,))))
c = jax.numpy.column_stack((a,a,-jax.numpy.ones((8*8,))))
print(c)