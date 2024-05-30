import jax

def calculate_pixel_coordinates(origin, lookat, width, height, fov):
    # Normalize the lookat vector
    lookat = lookat / jax.numpy.linalg.norm(lookat)

    # Calculate the up vector (assuming positive Y-axis)
    up = jax.numpy.array([0, 1, 0])

    # Calculate the right vector
    right = jax.numpy.cross(lookat, up)
    right = right / jax.numpy.linalg.norm(right)

    # Calculate the pixel size
    pixel_size = 2 * jax.numpy.tan(jax.numpy.radians(fov) / 2) / jax.numpy.maximum(width, height)

    def body_fn(carry, idx):
        y, x = jax.numpy.divmod(idx, width)
        u = (x + 0.5) / width
        v = (y + 0.5) / height

        px = (2 * u - 1) * (width / height) * jax.numpy.tan(jax.numpy.radians(fov) / 2)
        py = (1 - 2 * v) * jax.numpy.tan(jax.numpy.radians(fov) / 2)

        pixel_coord = origin + px * pixel_size * right + py * pixel_size * up + lookat
        return carry, pixel_coord

    _, pixel_coordinates = jax.lax.scan(body_fn, None, jax.numpy.arange(width * height))

    return pixel_coordinates

print(jax.jit(calculate_pixel_coordinates, static_argnames=["width", "height"])(jax.numpy.array([5.0, 10.0, 0.0]), jax.numpy.array([-1.0, 0.0, 1.0]), 64, 64, 60))