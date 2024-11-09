import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Charger le jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prétraiter les données (normalisation)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Redimensionner les données pour être compatibles avec les couches convolutives (ajouter une dimension de canaux)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Créer le modèle CNN
model = keras.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Sauvegarder le modèle dans le répertoire spécifié
model.save('C:/Users/bmond/OneDrive/GitHub/AI_Compression/my_model.h5')

print("Le modèle a été entraîné et sauvegardé sous my_model.h5.")
