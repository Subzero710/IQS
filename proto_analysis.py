#!.venv/Scripts/python.exe
from prototype import laplacian_sharpness, sobel_sharpness, local_gradient_variance, fft_sharpness, wavelet_transform, dct_sharpness , mono_image_memory
import os
import pandas as pd
from tqdm import tqdm  # Progress bar
from DFLJPG import DFLJPG
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def get_image_paths(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                yield os.path.join(root, file)

def process_image(image_path):
    try:
        image= cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dfl_img = DFLJPG.load(image_path)
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

        if mask_ratio < 0.2:
            raise ValueError(f"❌ ERROR: Mask too small ({mask_ratio * 100:.2f}% of image) for {image_path}")

        laplacian_value = laplacian_sharpness(image,mask)
        sobel_value = sobel_sharpness(image,mask)
        local_gradient_value = local_gradient_variance(image,mask)
        fft_value = fft_sharpness(image,mask)
        wavelet_value = wavelet_transform(image,mask)
        dct_value = dct_sharpness(image,mask)
        memory_value = mono_image_memory(image_path)

        return {
            "Image_Name": os.path.basename(image_path),
            "Laplacian_Sharpness": laplacian_value,
            "Sobel_Sharpness": sobel_value,
            "Local_Gradient_Variance": local_gradient_value,
            "FFT_Sharpness": fft_value,
            "Wavelet_Transform": wavelet_value,
            "DCT_Sharpness": dct_value,
            "IMG_Memory" : memory_value,
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
            "IMG_Memory" : "Error",
        }

if __name__ == "__main__":
    folder = input("Enter the folder path containing the images: ")
    output_file = input("Enter the output CSV file name (e.g., results.csv): ")

    image_paths = list(get_image_paths(folder))

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Processing images"))

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False, sep=",", float_format="%.2f")

    print(f"Scores have been saved to {output_file}")
