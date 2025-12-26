#!.venv/Scripts/python.exe
import os
import traceback
import shutil  # For moving files to their categorized folders
import numpy as np  # Processing on CPU
import pywt  # Wavelet Transform
from scipy.fftpack import dct
from scipy.ndimage import laplace, sobel
import cv2  # OpenCV for image handling  # Required for image handling
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor
import gc  # Garbage collection
from DFLJPG import DFLJPG
import json


# Sharpness metrics functions
def laplacian_sharpness(image,mask):
    laplacian = laplace(image)
    masked_values = laplacian[mask == 1]
    variance = np.var(masked_values)
    return variance

def sobel_sharpness(image,mask):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    masked_values = gradient_magnitude[mask == 1]
    mean_value = np.mean(masked_values)
    return mean_value

def local_gradient_variance(image,mask):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    masked_values = gradient_magnitude[mask == 1]
    variance = np.var(masked_values)
    return variance

def fft_sharpness(image,mask):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    masked_values = magnitude_spectrum[mask == 1]
    mean_value = np.mean(masked_values)
    return mean_value

def wavelet_transform(image,mask):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    energy = np.abs(cH) + np.abs(cV) + np.abs(cD)
    H, W = energy.shape
    masked_values = energy[mask[:H, :W] == 1]
    mean_value = np.mean(masked_values)
    return mean_value

def dct_sharpness(image,mask):
    dct_transform = dct(dct(image.T, norm='ortho').T, norm='ortho')
    high_freq = dct_transform[1:, 1:]
    masked_values = high_freq[mask[1:, 1:] == 1]
    mean_value = np.mean(np.abs(masked_values))
    return mean_value
''' 
def wavelet_transform(image,mask):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    mean_value = np.mean(np.abs(cH) + np.abs(cV) + np.abs(cD))
    return mean_value

def dct_sharpness(image,mask):
    dct_transform = dct(dct(image.T, norm='ortho').T, norm='ortho')
    high_freq = dct_transform[1:, 1:]
    mean_value = np.mean(np.abs(high_freq))
    return mean_value
'''
def load_thresholds(config_file="thresholds.json"):
    try:
        with open(config_file, "r") as f:
            thresholds = json.load(f)
        print(f"✅ Thresholds loaded from {config_file}")
        return thresholds
    except Exception as e:
        print(f"⚠️ Could not load thresholds from {config_file}, Error: {e}")

SCORING_THRESHOLDS = load_thresholds("thresholds.json")

def memory_size_score(image_size):
    thresholds = SCORING_THRESHOLDS["IMG_Memory"]
    if image_size < thresholds["min"]:
        return 0
    elif image_size <= thresholds["max"]:
        return ((image_size - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_laplacian(value):
    thresholds = SCORING_THRESHOLDS["Laplacian_Sharpness"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_sobel(value):
    thresholds = SCORING_THRESHOLDS["Sobel_Sharpness"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_local_gradient(value):
    thresholds = SCORING_THRESHOLDS["Local_Gradient_Variance"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_fft(value):
    thresholds = SCORING_THRESHOLDS["FFT_Sharpness"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_wavelet(value):
    thresholds = SCORING_THRESHOLDS["Wavelet_Transform"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

def score_dct(value):
    thresholds = SCORING_THRESHOLDS["DCT_Sharpness"]
    if value < thresholds["min"]:
        return 0
    elif value <= thresholds["max"]:
        return ((value - thresholds["min"]) / (thresholds["max"] - thresholds["min"])) * 1000
    else:
        return 1000

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
        file_size = os.path.getsize(image_path)
        return file_size
    except Exception as e:
        print(f"size_error for {image_path}: {e}")
        return 0

def load_images_batch(image_paths, max_buffer_size):
    images = {}
    current_buffer_size = 0

    for image_path in image_paths:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            image_array = img.astype(np.float32)  # Convert to NumPy array
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
        image_init = images.get(image_path)
        dfl_img = DFLJPG.load(image_path)
        image = image_init.copy()
        if image is None:
            raise ValueError("Image could not be loaded")
        if dfl_img is None:
            raise ValueError("dfl_img could not be loaded")
        mask = dfl_img.get_xseg_mask() if dfl_img and dfl_img.has_xseg_mask() else None
        mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        image = image * mask     #apply mask

        mask_area = np.sum(mask > 0)
        image_area = mask.shape[0] * mask.shape[1]
        mask_ratio = mask_area / image_area

        if mask_ratio < 0.15:
            raise ValueError(f"❌ ERROR: Mask too small ({mask_ratio * 100:.2f}% of image) for {image_path}")

        image_memory_size = mono_image_memory(image_path)
        laplacian_value = laplacian_sharpness(image,mask)
        sobel_value = sobel_sharpness(image,mask)
        local_gradient_value = local_gradient_variance(image,mask)
        fft_value = fft_sharpness(image,mask)
        wavelet_value = wavelet_transform(image,mask)
        dct_value = dct_sharpness(image,mask)

        laplacian_score = score_laplacian(laplacian_value)
        memory_score = memory_size_score(image_memory_size)
        sobel_score = score_sobel(sobel_value)
        local_gradient_score = score_local_gradient(local_gradient_value)
        fft_score= score_fft(fft_value)
        wavelet_score = score_wavelet(wavelet_value)
        dct_score= score_dct(dct_value)

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
            "IMG_Memory": image_memory_size,
            "Score_Category": category
        }
    except Exception as e:
        print(f"Error : {e}")
        #print(traceback.format_exc())
        return {
            "Image_Name": os.path.basename(image_path),
            "Laplacian_Sharpness": f"{e}",
            "Sobel_Sharpness": f"{e}",
            "Local_Gradient_Variance": f"{e}",
            "FFT_Sharpness": f"{e}",
            "Wavelet_Transform": f"{e}",
            "DCT_Sharpness": f"{e}",
            "IMG_Memory" : f"{e}",
            "Score_Category": "unclassified"
        }

def move_image_to_category(image_path, result, output_folder):
    category_folder = os.path.join(output_folder, result["Score_Category"])
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
