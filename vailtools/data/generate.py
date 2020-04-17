import numpy as np
from skimage import transform


def pickup_sticks_image(
    n_sticks: int = 15, lenth_range=(80, 150), width_range=(10, 20), image_dims=(256, 256, 3),
):
    """
    Generates a RGB image containing randomly created "sticks" on a uniform background.
    The sticks have varying colors, shapes, positions, and rotations within the image.
    The sticks may overlap, leading to visual occlusion.
    Also provides a set of ground truth labels that indicate the location of each stick.

    Args:
        n_sticks: The number of objects to create.
        lenth_range: Bounds on stick length (in pixels)
        width_range: Bounds on stick width (in pixels)
        image_dims: Desired dimensions of the generated image

    Returns: Tuple(numpy.ndarray, numpy.ndarray)
        An RGB image and corresponding label.
    """
    image = np.zeros(image_dims)
    labels = np.zeros(image_dims)

    # Set a random background color
    image[:] = np.random.random(size=(1, 1, 3))
    for i in range(n_sticks):
        # Generate random stick params
        stick_length = np.random.randint(*lenth_range)
        stick_width = np.random.randint(*width_range)
        stick_pos = (np.random.randint(image_dims[0]), np.random.randint(image_dims[1]))
        stick_rotation = np.pi * (2 * np.random.random() - 1)
        stick_color = np.random.random(size=(1, 1, 3))

        # Add stick to image
        stick_image = np.zeros(image_dims)
        stick_image[:stick_length, :stick_width] = stick_color

        # Transform image
        stick_transform = transform.AffineTransform(
            rotation=stick_rotation, translation=stick_pos
        )
        stick_image_trans = transform.warp(stick_image, stick_transform.inverse)

        # Add new stick to composite image
        inds = np.where(stick_image_trans != 0)
        image[inds] = stick_image_trans[inds]
        labels[inds] = i + 1

    # (height, width, channels) -> (height, width, 1)
    labels = labels.max(axis=-1, keepdims=True)
    return image, labels


if __name__ == "__main__":
    # Visual confirmation of image generation
    import matplotlib.pyplot as plt

    image, labels = pickup_sticks_image()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 9.6))
    axes[0].imshow(image, origin="lower")
    axes[1].imshow(labels.squeeze() / labels.max(), origin="lower")
    plt.show()
    plt.close("all")
