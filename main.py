import argparse

import pandas as pd

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

FEATURES = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn client")
    parser.add_argument("--data", default="Churn_Modelling.csv", help="Chemin du CSV")
    parser.add_argument("--model", default="classifier.joblib", help="Chemin du modèle")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner et sauvegarder")
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    parser.add_argument(
        "--predict",
        nargs=len(FEATURES),
        type=float,
        metavar="VAL",
        help="Prédire pour un client : " + " ".join(FEATURES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Sans option : exécution du pipeline complet (prepare -> train -> evaluate)
    if not any([args.prepare, args.train, args.evaluate, args.predict]):
        args.prepare = args.train = args.evaluate = True

    if args.prepare or args.train or args.evaluate:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        if args.prepare:
            print(f"Données prêtes : train={X_train.shape}, test={X_test.shape}")

    if args.train:
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    if args.evaluate:
        evaluate_model(load_model(args.model), X_test, y_test)

    if args.predict:
        model = load_model(args.model)
        sample = pd.DataFrame([args.predict], columns=FEATURES)
        label = int(model.predict(sample)[0])
        print(
            "Prédiction :",
            "Exited (client perdu)" if label else "Not Exited (client conservé)",
        )


if __name__ == "__main__":
    main()
