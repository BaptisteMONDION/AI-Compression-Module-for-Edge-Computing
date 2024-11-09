#!/bin/bash

# Définir les chemins des répertoires pour les fichiers nécessaires
PROJECT_DIR="C:/Users/bmond/OneDrive/GitHub/AI_Compression"
MODEL_DIR="$PROJECT_DIR/models"
MODEL_PATH="$MODEL_DIR/my_model.h5"
COMPRESSED_MODEL_PATH="$MODEL_DIR/compressed_model.h5"

# Vérifier si TensorFlow est installé
check_tensorflow_version() {
    echo "Vérification de la version de TensorFlow..."
    TF_VERSION=$(python -c 'import tensorflow as tf; print(tf.__version__)')
    echo "Version actuelle de TensorFlow: $TF_VERSION"
    if [[ "$TF_VERSION" < "2.0.0" ]]; then
        echo "TensorFlow version $TF_VERSION est trop ancienne. Assurez-vous d'avoir TensorFlow 2.x installé."
        exit 1
    else
        echo "TensorFlow est correctement installé avec la version $TF_VERSION"
    fi
}

# Installer TensorFlow (si non installé)
install_tensorflow() {
    echo "Installation de TensorFlow..."
    pip install --upgrade tensorflow
}

# Installer les dépendances nécessaires
install_dependencies() {
    echo "Installation des dépendances nécessaires..."
    pip install tensorflow tensorflow-model-optimization
}

# Vérifier si les chemins existent
check_paths() {
    echo "Vérification des chemins de répertoires..."

    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Le répertoire du projet n'existe pas : $PROJECT_DIR"
        exit 1
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Le modèle n'existe pas à l'emplacement spécifié : $MODEL_PATH"
        exit 1
    fi

    echo "Tous les chemins sont valides."
}

# Fonction principale
main() {
    # Vérifier les chemins
    check_paths

    # Vérifier la version de TensorFlow
    check_tensorflow_version

    # Installer TensorFlow (si nécessaire)
    install_tensorflow

    # Installer les dépendances
    install_dependencies

    # Afficher un message de fin
    echo "Configuration terminée avec succès ! Vous pouvez maintenant exécuter votre module de compression IA."
}

# Lancer la fonction principale
main
