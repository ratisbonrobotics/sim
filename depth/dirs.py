import numpy as np

def calculate_pixel_coordinates(origin, lookat, width, height, fov):
    # Normalize the lookat vector
    lookat = lookat / np.linalg.norm(lookat)

    # Calculate the up vector (assuming positive Y-axis)
    up = np.array([0, 1, 0])

    # Calculate the right vector
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)

    # Calculate the pixel size
    pixel_size = 2 * np.tan(np.radians(fov) / 2) / max(width, height)

    pixel_coordinates = []

    for y in range(height):
        for x in range(width):
            u = (x + 0.5) / width
            v = (y + 0.5) / height

            px = (2 * u - 1) * (width / height) * np.tan(np.radians(fov) / 2)
            py = (1 - 2 * v) * np.tan(np.radians(fov) / 2)

            pixel_coord = origin + px * pixel_size * right + py * pixel_size * up + lookat
            pixel_coordinates.append(pixel_coord)

    return pixel_coordinates

print(calculate_pixel_coordinates([5.0,10.0,0.0], [-1.0,0.0,1.0], 64, 64, 60))