import os
import numpy as np  # Processing on CPU
import pandas as pd
from PIL import Image  # Required for image handling
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor
import gc  # Garbage collection

# Exposure metrics functions
def histogram_spread(image):
    spread = image.max() - image.min()
    return spread

def mean_absolute_error(image):
    mae = np.mean(np.abs(image - np.mean(image)))
    return mae

def mean_squared_error(image):
    mse = np.mean((image - np.mean(image))**2)
    return mse

def variance_intensity(image):
    variance = np.var(image)
    return variance

# Function to get all image paths in a folder
def get_image_paths(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                yield os.path.join(root, file)

def estimate_image_memory(image_paths):
    try:
        sample_image = Image.open(image_paths[0]).convert('L')
        sample_array = np.array(sample_image, dtype=np.float64)
        return sample_array.nbytes
    except Exception as e:
        return 1024 * 1024  # Default to 1MB if estimation fails

def load_images_batch(image_paths, max_buffer_size):
    images = {}
    current_buffer_size = 0

    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            image_array = np.array(img, dtype=np.float64)  # Convert to NumPy array
            image_size = image_array.nbytes

            if current_buffer_size + image_size > max_buffer_size:
                break

            images[image_path] = image_array
            current_buffer_size += image_size
        except Exception as e:
            images[image_path] = None

    return images

def process_image(image_path, images):
    try:
        image = images.get(image_path)
        if image is None:
            raise ValueError("Image could not be loaded")

        return {
            "Image_Name": os.path.basename(image_path),
            "Histogram_Spread": round(float(histogram_spread(image)), 2),
            "MAE": round(float(mean_absolute_error(image)), 2),
            "MSE": round(float(mean_squared_error(image)), 2),
            "Variance_Intensity": round(float(variance_intensity(image)), 2)
        }
    except Exception as e:
        return {
            "Image_Name": os.path.basename(image_path),
            "Histogram_Spread": "Error",
            "MAE": "Error",
            "MSE": "Error",
            "Variance_Intensity": "Error"
        }

# Main entry point
if __name__ == "__main__":
    folder = input("Enter the folder path containing the images: ")
    max_buffer_size = int(input("Enter the maximum buffer size in GB (e.g., 10): ")) * 1024 * 1024 * 1024

    exposure_results = []
    image_paths = list(get_image_paths(folder))
    image_memory_size = estimate_image_memory(image_paths)
    batch_size = max(1, max_buffer_size // image_memory_size)

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")

        images = load_images_batch(batch, max_buffer_size)

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda p: process_image(p, images), batch), total=len(batch), desc="Processing images"))
            exposure_results.extend(results)

        # Free memory
        del images
        gc.collect()

    df_exposure_results = pd.DataFrame(exposure_results)

    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "exposure_results.csv")
    df_exposure_results.to_csv(output_file, index=False, sep=",", float_format="%.2f")
    print(f"Exposure results saved to: {output_file}")
