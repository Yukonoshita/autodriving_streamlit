from PIL import Image
import os

def create_dataset_overview(image_folder, output_path, images_per_row, target_size=(256, 256), margin=10):
    """
    Creates an overview image from a collection of images with specified margins between them,
    resizing them to a uniform size.

    :param image_folder: Path to the folder containing images.
    :param output_path: Path where the combined image will be saved.
    :param images_per_row: Number of images in each row.
    :param target_size: Tuple (width, height) specifying the new size for all images.
    :param margin: Space between images in pixels.
    """
    # Gather all images from the folder
    image_files = [f for f in os.listdir(image_folder)]
    image_files.sort()

    # Open images and resize them
    images = [Image.open(os.path.join(image_folder, img)).resize(target_size, Image.ANTIALIAS) for img in image_files]

    # Calculate total dimensions of the final image
    total_width = target_size[0] * images_per_row + margin * (images_per_row - 1)
    total_rows = len(images) // images_per_row + (1 if len(images) % images_per_row else 0)
    total_height = target_size[1] * total_rows + margin * (total_rows - 1)

    # Create a new blank white image to place the overview
    new_im = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images into the new image
    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        new_im.paste(img, (x_offset, y_offset))
        x_offset += target_size[0] + margin
        if (i + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += target_size[1] + margin

    # Save the new image
    new_im.save(output_path)

# Usage example
# image_folder = 'E:/Project/Zero-DCE/Zero-DCE_code/data/test_data/LIME'  # Folder containing images
image_folder = './test'  # Folder containing images

output_path = 'dataset_overview2.jpg'  # Path to save the overview image
images_per_row = 5  # Number of images per row in the overview
target_size = (256, 256)  # Desired size for each image in the overview
margin = 10  # Margin between images

create_dataset_overview(image_folder, output_path, images_per_row, target_size, margin)
