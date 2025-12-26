import cupy as cp  # GPU processing with CuPy
import pandas as pd
import os
import pywt  # Wavelet Transform
from scipy.fftpack import dct
from cupyx.scipy.ndimage import laplace, sobel
import numpy as np
from tqdm import tqdm  # Progress bar
from cupy.cuda import stream
from concurrent.futures import ThreadPoolExecutor
from PIL import Image  # Ensure Pillow is imported
import imageio.v3 as iio  # Efficient image loading

# Functions for sharpness metrics remain unchanged
def laplacian_sharpness(image):
    variance = laplace(image).var()
    return variance

def sobel_sharpness(image):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = cp.sqrt(sobel_x**2 + sobel_y**2)
    mean_value = gradient_magnitude.mean()
    return mean_value

def local_gradient_variance(image):
    sobel_x = sobel(image, axis=0)
    sobel_y = sobel(image, axis=1)
    gradient_magnitude = cp.sqrt(sobel_x**2 + sobel_y**2)
    variance = cp.var(gradient_magnitude)
    return variance

def fft_sharpness(image):
    f_transform = cp.fft.fft2(image)
    f_shift = cp.fft.fftshift(f_transform)
    magnitude_spectrum = cp.abs(f_shift)
    mean_value = cp.mean(magnitude_spectrum)
    return mean_value

def wavelet_transform(image):
    coeffs = pywt.dwt2(cp.asnumpy(image), 'haar')
    cA, (cH, cV, cD) = coeffs
    cH_gpu = cp.asarray(cH)
    cV_gpu = cp.asarray(cV)
    cD_gpu = cp.asarray(cD)
    mean_value = cp.mean(cp.abs(cH_gpu) + cp.abs(cV_gpu) + cp.abs(cD_gpu))
    return mean_value

def variance_intensity(image):
    variance = cp.var(image)
    return variance

def dct_sharpness(image):
    dct_transform = dct(dct(cp.asnumpy(image).T, norm='ortho').T, norm='ortho')
    high_freq = dct_transform[1:, 1:]
    mean_value = cp.mean(cp.abs(cp.asarray(high_freq)))
    return mean_value

def histogram_spread(image):
    spread = image.max() - image.min()
    return spread

def mean_absolute_error(image):
    mae = cp.mean(cp.abs(image - cp.mean(image)))
    return mae

def mean_squared_error(image):
    mse = cp.mean((image - cp.mean(image))**2)
    return mse

def load_image_to_cupy(image_path):
    img = iio.imread(image_path, mode="L")  # Load grayscale image directly
    return cp.array(img, dtype=cp.float64)

def get_image_paths(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                yield os.path.join(root, file)

# Function to process a batch of images using a specific CUDA stream
def process_batch_with_stream(image_paths, results, stream_obj):
    with stream_obj:
        for image_path in image_paths:
            try:
                image = load_image_to_cupy(image_path)

                laplacian_value = laplacian_sharpness(image)
                sobel_value = sobel_sharpness(image)
                local_gradient_value = local_gradient_variance(image)
                fft_value = fft_sharpness(image)
                histogram_value = histogram_spread(image)
                wavelet_value = wavelet_transform(image)
                variance_value = variance_intensity(image)
                dct_value = dct_sharpness(image)
                mae_value = mean_absolute_error(image)
                mse_value = mean_squared_error(image)

                results["Image_Name"].append(os.path.basename(image_path))
                cp.get_default_memory_pool().free_all_blocks()  # Free memory after processing each image
                results["Laplacian_Sharpness"].append(round(float(laplacian_value), 2))
                results["Sobel_Sharpness"].append(round(float(sobel_value), 2))
                results["Local_Gradient_Variance"].append(round(float(local_gradient_value), 2))
                results["FFT_Sharpness"].append(round(float(fft_value), 2))
                results["Histogram_Spread"].append(round(float(histogram_value), 2))
                results["Wavelet_Transform"].append(round(float(wavelet_value), 2))
                results["Variance_Intensity"].append(round(float(variance_value), 2))
                results["DCT_Sharpness"].append(round(float(dct_value), 2))
                results["MAE"].append(round(float(mae_value), 2))
                results["MSE"].append(round(float(mse_value), 2))
            except Exception as e:
                results["Image_Name"].append(os.path.basename(image_path))
                for key in results.keys():
                    if key != "Image_Name":
                        results[key].append("Error")
        stream_obj.synchronize()
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after synchronization

if __name__ == "__main__":
    if cp.cuda.is_available():
        print("Using GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'])
    else:
        print("No compatible GPU detected.")
        exit()

    folder = input("Enter the folder path containing the images: ")
    batch_size = int(input("Enter the batch size: "))
    num_threads = int(input("Enter the number of threads: "))

    sharpness_results = {
        "Image_Name": [],
        "Laplacian_Sharpness": [],
        "Sobel_Sharpness": [],
        "Local_Gradient_Variance": [],
        "FFT_Sharpness": [],
        "Histogram_Spread": [],
        "Wavelet_Transform": [],
        "Variance_Intensity": [],
        "DCT_Sharpness": [],
        "MAE": [],
        "MSE": []
    }

    image_paths = list(get_image_paths(folder))

    with tqdm(total=len(image_paths), desc="Processing images") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                stream_obj = stream.Stream()  # Create a new stream for each batch
                futures.append(executor.submit(process_batch_with_stream, batch, sharpness_results, stream_obj))

            for future in futures:
                future.result()  # Wait for all threads to complete
                pbar.update(batch_size)

    df_sharpness_results = pd.DataFrame(sharpness_results)

    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "sharpness_results.csv")
    df_sharpness_results.to_csv(output_file, index=False, sep=",", float_format="%.2f")
    print(f"Sharpness results saved to: {output_file}")
