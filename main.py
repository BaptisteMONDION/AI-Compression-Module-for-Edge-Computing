import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# Définir le répertoire du projet et du modèle
PROJECT_DIR = "C:/Users/bmond/OneDrive/GitHub/AI_Compression"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.h5")
COMPRESSED_MODEL_PATH = os.path.join(MODEL_DIR, "compressed_model.h5")

# Vérifier si TensorFlow est installé
def check_tensorflow_version():
    print("Vérification de la version de TensorFlow...")
    tf_version = tf.__version__
    print(f"Version actuelle de TensorFlow: {tf_version}")
    if tf_version < "2.0.0":
        print(f"TensorFlow version {tf_version} est trop ancienne. Assurez-vous d'avoir TensorFlow 2.x installé.")
        exit(1)
    else:
        print(f"TensorFlow est correctement installé avec la version {tf_version}")

# Charger le modèle
def load_model():
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Le modèle n'existe pas à l'emplacement spécifié : {MODEL_PATH}")
        exit(1)
    return keras.models.load_model(MODEL_PATH)

# Appliquer la compression du modèle
def compress_model(model):
    print("Application de la compression...")
    # Exemple de compression via pruning (réduction des poids)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model_for_pruning = prune_low_magnitude(model)
    return model_for_pruning

# Sauvegarder le modèle compressé
def save_compressed_model(model):
    print(f"Enregistrement du modèle compressé à {COMPRESSED_MODEL_PATH}...")
    model.save(COMPRESSED_MODEL_PATH)

# Fonction principale
def main():
    # Vérifier la version de TensorFlow
    check_tensorflow_version()

    # Charger et compresser le modèle
    model = load_model()
    compressed_model = compress_model(model)

    # Sauvegarder le modèle compressé
    save_compressed_model(compressed_model)
    print("Compression terminée avec succès!")

if __name__ == "__main__":
    main()
