#!.venv/Scripts/python.exe
import pandas as pd
import numpy as np
import json

# Charger le CSV
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSV chargé avec {len(df)} entrées.")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement du CSV : {e}")
        return None


# Calcul des seuils basés sur les percentiles
def calculate_thresholds(df):
    thresholds = {}

    metrics = [
        "Laplacian_Sharpness", "Sobel_Sharpness", "Local_Gradient_Variance",
        "FFT_Sharpness", "Wavelet_Transform", "DCT_Sharpness", "IMG_Memory"
    ]

    for metric in metrics:
        if metric in df.columns:
            # Supprimer les valeurs "Error" et convertir en float
            values = pd.to_numeric(df[metric], errors='coerce').dropna().values

            if len(values) > 0:
                min_threshold = np.percentile(values, 5)  # 5ᵉ percentile
                max_threshold = np.percentile(values, 95)  # 95ᵉ percentile
                thresholds[metric] = {"min": float(min_threshold), "max": float(max_threshold)}
                print(f"📊 {metric}: min={min_threshold:.2f}, max={max_threshold:.2f}")
            else:
                print(f"⚠️ {metric} n'a pas assez de données pour calculer les seuils.")
        else:
            print(f"⚠️ Colonne {metric} introuvable dans le CSV.")

    return thresholds

def save_thresholds(thresholds, json_file):
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=4)
        print(f"✅ Seuils sauvegardés dans {json_file}.")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du JSON : {e}")



if __name__ == "__main__":
    CSV_FILE = input("enter analysis csv path :")
    JSON_FILE = "thresholds.json"
    df = load_csv(CSV_FILE)
    if df is not None:
        thresholds = calculate_thresholds(df)
        save_thresholds(thresholds, JSON_FILE)
