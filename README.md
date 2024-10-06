# Modèle d'analyse de la méthylation de l'ADN

Ce projet utilise un réseau neuronal convolutionnel (Conv1D) pour prédire la méthylation de l'ADN basée sur des séquences génomiques encodées.

## Structure du projet

- `data_preprocessing.py` : Contient le code pour charger les données brutes, les encoder en vecteurs et effectuer le padding pour uniformiser la longueur des séquences.
- `model.py` : Définit l'architecture du modèle neuronal avec des couches convolutionnelles, de pooling, et de régularisation.
- `train_model.py` : Entraîne le modèle avec les données encodées et affiche les courbes de performance (précision et perte) en fonction des époques.
- `README.md` : Documentation du projet.

## Prérequis

Assurez-vous d'avoir les bibliothèques suivantes installées :

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Installez-les avec pip si nécessaire :

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
