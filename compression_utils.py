import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# Définir les chemins du projet et du modèle
PROJECT_DIR = "C:/Users/bmond/OneDrive/GitHub/AI_Compression"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.h5")
COMPRESSED_MODEL_PATH = os.path.join(MODEL_DIR, "compressed_model.h5")

# Appliquer la compression au modèle
def apply_compression(model):
    print("Compression en cours...")
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    compressed_model = prune_low_magnitude(model)
    return compressed_model

# Sauvegarder le modèle compressé
def save_compressed_model(model):
    print(f"Enregistrement du modèle compressé à {COMPRESSED_MODEL_PATH}...")
    model.save(COMPRESSED_MODEL_PATH)

# Charger le modèle et appliquer la compression
def load_and_compress_model():
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Le modèle n'existe pas à l'emplacement spécifié : {MODEL_PATH}")
        exit(1)
    model = keras.models.load_model(MODEL_PATH)
    compressed_model = apply_compression(model)
    save_compressed_model(compressed_model)

if __name__ == "__main__":
    load_and_compress_model()
