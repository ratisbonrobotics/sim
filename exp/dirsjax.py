import jax

def calculate_pixel_coordinates(origin, lookat, width, height, fov):
    lookat = lookat / jax.numpy.linalg.norm(lookat)
    up = jax.numpy.array([0, 1, 0])
    right = jax.numpy.cross(lookat, up)
    right = right / jax.numpy.linalg.norm(right)
    pixel_size = 2 * jax.numpy.tan(jax.numpy.radians(fov) / 2) / jax.numpy.maximum(width, height)

    def yloop(x, carry):
        (pixel_lookats, pixel_coordinates) = carry
        u = (x + 0.5) / width
        px = (2 * u - 1) * (width / height) * jax.numpy.tan(jax.numpy.radians(fov) / 2)

        def xloop(y, carry):
            (pixel_lookats, pixel_coordinates) = carry
            v = (y + 0.5) / height
            py = (1 - 2 * v) * jax.numpy.tan(jax.numpy.radians(fov) / 2)
            pixel_coord = origin + px * pixel_size * right + py * pixel_size * up + lookat
            pixel_coordinates = pixel_coordinates.at[y * width + x].set(pixel_coord)
            pixel_lookats = pixel_lookats.at[y * width + x].set(pixel_coord + lookat)
            return (pixel_lookats, pixel_coordinates)

        pixel_lookats, pixel_coordinates = jax.lax.fori_loop(0, height, xloop, (pixel_lookats, pixel_coordinates))
        return (pixel_lookats, pixel_coordinates)

    pixel_coordinates = jax.numpy.zeros((height * width, 3))
    pixel_lookats = jax.numpy.zeros((height * width, 3))
    pixel_lookats, pixel_coordinates = jax.lax.fori_loop(0, width, yloop, (pixel_lookats, pixel_coordinates))

    return (pixel_lookats, pixel_coordinates)

pixel_lookats, pixel_coordinates = jax.jit(calculate_pixel_coordinates, static_argnames=["width", "height"])(jax.numpy.array([5.0, 10.0, 0.0]), jax.numpy.array([-1.0, 0.0, 1.0]), 64, 64, 60)

print(pixel_coordinates)
print(pixel_lookats)

