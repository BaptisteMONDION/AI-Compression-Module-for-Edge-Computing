import tensorflow as tf
import os
from tensorflow import keras

# Définir les chemins du projet et du modèle
PROJECT_DIR = "C:/Users/bmond/OneDrive/GitHub/AI_Compression"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.h5")

# Vérifier la version de TensorFlow
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
