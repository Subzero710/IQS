#!.venv/Scripts/python.exe
from sharpness_analysis import laplacian_sharpness, sobel_sharpness, local_gradient_variance, fft_sharpness, wavelet_transform, dct_sharpness
import os
import numpy as np  # Processing on CPU
import pandas as pd
from PIL import Image  # Required for image handling
from tqdm import tqdm  # Progress bar

def get_image_paths(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                yield os.path.join(root, file)

def process_image(image_path):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        image = np.array(img, dtype=np.float64)  # Convert to NumPy array
        laplacian_value = laplacian_sharpness(image)
        sobel_value = sobel_sharpness(image)
        local_gradient_value = local_gradient_variance(image)
        fft_value = fft_sharpness(image)
        wavelet_value = wavelet_transform(image)
        dct_value = dct_sharpness(image)

        return {
            "Image_Name": os.path.basename(image_path),
            "Laplacian_Sharpness": laplacian_value,
            "Sobel_Sharpness": sobel_value,
            "Local_Gradient_Variance": local_gradient_value,
            "FFT_Sharpness": fft_value,
            "Wavelet_Transform": wavelet_value,
            "DCT_Sharpness": dct_value,
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
            "BRISQUE": "Error"
        }

if __name__ == "__main__":
    folder = input("Enter the folder path containing the images: ")
    output_file = input("Enter the output CSV file name (e.g., results.csv): ")

    image_paths = list(get_image_paths(folder))
    results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        result = process_image(image_path)
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False, sep=",", float_format="%.2f")

    print(f"Scores have been saved to {output_file}")
