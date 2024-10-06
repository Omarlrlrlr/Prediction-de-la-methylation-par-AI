import numpy as np
import pandas as pd

# Charger les données
data = pd.read_csv('new.csv')  # Remplace par le chemin de ton fichier

# Fonction pour encoder les séquences ADN (inclut le handling de 'N')
def encode_sequence(seq):
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # N est encodé avec un vecteur de zéros
    }
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq])

# Appliquer l'encodage à la colonne des séquences
data['encoded_sequence'] = data['Séquence'].apply(encode_sequence)

# Padding des séquences pour avoir la même longueur
max_length = max(data['encoded_sequence'].apply(len))
data['encoded_sequence'] = data['encoded_sequence'].apply(lambda x: np.pad(x, ((0, max_length - len(x)), (0, 0)), mode='constant'))

# Préparer les features et les labels
X = np.stack(data['encoded_sequence'].values)
y = data['Méthylation'].apply(lambda x: 1 if x == 'Z' else 0).values

# Sauvegarder les features et les labels dans des fichiers séparés
np.save('X.npy', X)
np.save('y.npy', y)
