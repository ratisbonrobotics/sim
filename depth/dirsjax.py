import jax
import jax.numpy as jnp

def calculate_pixel_coordinates(origin, lookat, width, height, fov):
    # Normalize the lookat vector
    lookat = lookat / jnp.linalg.norm(lookat)

    # Calculate the up vector (assuming positive Y-axis)
    up = jnp.array([0, 1, 0])

    # Calculate the right vector
    right = jnp.cross(lookat, up)
    right = right / jnp.linalg.norm(right)

    # Calculate the pixel size
    pixel_size = 2 * jnp.tan(jnp.radians(fov) / 2) / jnp.maximum(width, height)

    def body_fn(carry, idx):
        y, x = jnp.divmod(idx, width)
        u = (x + 0.5) / width
        v = (y + 0.5) / height

        px = (2 * u - 1) * (width / height) * jnp.tan(jnp.radians(fov) / 2)
        py = (1 - 2 * v) * jnp.tan(jnp.radians(fov) / 2)

        pixel_coord = origin + px * pixel_size * right + py * pixel_size * up + lookat
        return carry, pixel_coord

    _, pixel_coordinates = jax.lax.scan(body_fn, None, jnp.arange(width * height))

    return pixel_coordinates

print(jax.jit(calculate_pixel_coordinates, static_argnames=["width", "height"])(jnp.array([5.0, 10.0, 0.0]), jnp.array([-1.0, 0.0, 1.0]), 64, 64, 60))