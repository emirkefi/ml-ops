import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
# Colonnes inutiles pour la prédiction (identifiants, texte libre, catégorie non encodée)
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, prétraite les données et les sépare en train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité du split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET}' absente du fichier.")

    # Encodage de Gender (Female=0, Male=1)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    X = df.drop(columns=[TARGET] + COLUMNS_TO_DROP, errors="ignore")
    y = df[TARGET]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier.

    Args:
        X_train (DataFrame): features d'entraînement.
        y_train (Series): cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine aléatoire.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test.

    Returns:
        dict: accuracy, matrice de confusion et rapport de classification.
    """
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    print(f"Accuracy : {results['accuracy'] * 100:.2f} %")
    print("Matrice de confusion :")
    print(results["confusion_matrix"])
    print(results["classification_report"])
    return results


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path} (lancez d'abord --train)")
    return joblib.load(path)
