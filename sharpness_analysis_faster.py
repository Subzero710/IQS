#!.venv/Scripts/python.exe
import sys  # For modifying path to include BRISQUE dependencies
# Append LIBSVM path if needed
sys.path.append("C:/Users/FlowUP/PycharmProjects/IQS/.venv/Lib/site-packages/libsvm")
import os
import shutil  # For moving files to their categorized folders
import numpy as np  # Processing on CPU
#import pandas as pd
import pywt  # Wavelet Transform
from scipy.fftpack import dct
from scipy.ndimage import laplace, sobel
import cv2  # OpenCV for image handling  # Required for image handling
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor
import gc  # Garbage collection
#from brisque import BRISQUE  # For BRISQUE computation

#brisq = BRISQUE()  # Initialize BRISQUE once


# Sharpness metrics functions
def laplacian_sharpness(image):
    variance = laplace(image).var()
    return variance

def sobel_sharpness(image):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mean_value = gradient_magnitude.mean()
    return mean_value

def local_gradient_variance(image):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    variance = np.var(gradient_magnitude)
    return variance

def fft_sharpness(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    mean_value = np.mean(magnitude_spectrum)
    return mean_value

def wavelet_transform(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    mean_value = np.mean(np.abs(cH) + np.abs(cV) + np.abs(cD))
    return mean_value

def dct_sharpness(image):
    dct_transform = dct(dct(image.T, norm='ortho').T, norm='ortho')
    high_freq = dct_transform[1:, 1:]
    mean_value = np.mean(np.abs(high_freq))
    return mean_value

def memory_size_score(image_size, vouched):
    """
    Score based on image memory size in bytes.
    """
    # Define thresholds for scoring
    if image_size < 1024 * 300:  # Less than 300 KB
        return 0, vouched
    elif image_size <= 1024 * 400:  # Between 300 KB and 400 KB
        return ((image_size - 1024 * 300) / (1024 * 100)) * 1000, vouched
    else:  # Greater than 400 KB
        vouched += 1
        return 1000, vouched

# Scoring functions
def score_laplacian(value, vouched):
    if value < 10:
        return 0, vouched
    elif value <= 200:
        return ((value - 10) / 190) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

'''def score_brisque(value, vouched):
    if value < 10:
        return 0, vouched
    elif value <= 90:
        return ((value - 10) / 90) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched
'''
def score_sobel(value, vouched):
    if value < 15:
        return 0, vouched
    elif value <= 40:
        return ((value - 15) / 25) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

def score_local_gradient(value, vouched):
    if value < 600:
        return 0, vouched
    elif value <= 5000:
        return ((value - 600) / 4400) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

def score_fft(value, vouched):
    if value < 2000:
        return 0, vouched
    elif value <= 5000:
        return ((value - 2000) / 3000) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

def score_wavelet(value, vouched):
    if value < 2.5:
        return 0, vouched
    elif value <= 7.5:
        return ((value - 2.5) / 5) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

def score_dct(value, vouched):
    if value < 2.5:
        return 0, vouched
    elif value <= 7.5:
        return ((value - 2.5) / 5) * 1000, vouched
    else:
        vouched += 1
        return 1000, vouched

def get_score_category(score):
    if score > 900:
        return "perfect"
    elif score > 800:
        return "900-800"
    elif score > 700:
        return "800-700"
    elif score > 600:
        return "700-600"
    elif score > 500:
        return "600-500"
    else:
        return "trash"


# Function to get all image paths in a folder
def get_image_paths(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                yield os.path.join(root, file)

def estimate_image_memory(image_paths):
    try:
        sample_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        sample_array = np.array(sample_image, dtype=np.float64)
        return sample_array.nbytes
    except Exception as e:
        return 1024 * 1024  # Default to 1MB if estimation fails
def mono_image_memory(image_path):
    try:
        sample_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sample_array = np.array(sample_image, dtype=np.float64)
        return sample_array.nbytes
    except Exception as e:
        return 1024 * 1024  # Default to 1MB if estimation fails

def load_images_batch(image_paths, max_buffer_size):
    images = {}
    current_buffer_size = 0

    for image_path in image_paths:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            image_array = img.astype(np.float64)  # Convert to NumPy array
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
        vouched = 0
        image = images.get(image_path)
        if image is None:
            raise ValueError("Image could not be loaded")

        # Convert to uint8 for BRISQUE, as it requires this format
        #image_uint8 = (image / np.max(image) * 255).astype(np.uint8)
        image_memory_size = mono_image_memory(image_path)
        laplacian_value = laplacian_sharpness(image)
        sobel_value = sobel_sharpness(image)
        local_gradient_value = local_gradient_variance(image)
        fft_value = fft_sharpness(image)
        wavelet_value = wavelet_transform(image)
        dct_value = dct_sharpness(image)
        #brisque_value = brisq.get_score(image_uint8)

        laplacian_score, vouched = score_laplacian(laplacian_value, vouched)
        memory_score, vouched = memory_size_score(image_memory_size,vouched)
        sobel_score, vouched = score_sobel(sobel_value, vouched)
        local_gradient_score, vouched = score_local_gradient(local_gradient_value, vouched)
        fft_score, vouched = score_fft(fft_value, vouched)
        wavelet_score, vouched = score_wavelet(wavelet_value, vouched)
        dct_score, vouched = score_dct(dct_value, vouched)
        #brisque_score, vouched = score_brisque(brisque_value,vouched)

        total_score = ((laplacian_score + sobel_score + local_gradient_score + fft_score + wavelet_score + dct_score + memory_score ) / 7)
        category = get_score_category(total_score)

        return {
            "Image_Name": os.path.basename(image_path),
            "Laplacian_Sharpness": laplacian_value,
            "Sobel_Sharpness": sobel_value,
            "Local_Gradient_Variance": local_gradient_value,
            "FFT_Sharpness": fft_value,
            "Wavelet_Transform": wavelet_value,
            "DCT_Sharpness": dct_value,
            #"BRISQUE": brisque_value,
            "IMG_Memory": image_memory_size,
            "Vouched": vouched,
            "Score_Category": category
        }
    except Exception as e:
        return {
            "Image_Name": os.path.basename(image_path),
            "Laplacian_Sharpness": "Error",
            "Sobel_Sharpness": "Error",
            "Local_Gradient_Variance": "Error",
            "FFT_Sharpness": "Error",
            "Wavelet_Transform": "Error",
            "DCT_Sharpness": "Error",
            #"BRISQUE": "Error",
            "IMG_Memory" : "Error",
            "Vouched": 0,
            "Score_Category": "unclassified"
        }

def move_image_to_category(image_path, result, output_folder):
    vouched_folder = os.path.join(output_folder, "vouched_by" if result["Vouched"] > 0 else "non_vouched")
    vouched_subfolder = os.path.join(vouched_folder, str(result["Vouched"])) if result["Vouched"] > 0 else vouched_folder
    category_folder = os.path.join(vouched_subfolder, result["Score_Category"])

    os.makedirs(category_folder, exist_ok=True)
    dest_path = os.path.join(category_folder, result["Image_Name"])
    shutil.move(image_path, dest_path)


# Main entry point
if __name__ == "__main__":
    folder = input("Enter the folder path containing the images: ")
    output_folder = "Result"
    max_buffer_size = int(input("Enter the maximum buffer size in GB (e.g., 10): ")) * 1024 * 1024 * 1024

    image_paths = list(get_image_paths(folder))
    image_memory_size = estimate_image_memory(image_paths)
    batch_size = max(1, max_buffer_size // image_memory_size)

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")

        images = load_images_batch(batch, max_buffer_size)

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda p: process_image(p, images), batch), total=len(batch),
                                desc="Processing images"))

            for image_path, result in zip(batch, results):
                move_image_to_category(image_path, result, output_folder)

        # Free memory
        del images
        gc.collect()

    print(f"All images have been processed and moved to {output_folder}")
