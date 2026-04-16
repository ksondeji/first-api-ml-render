"""Entraîne un classifieur minimal (2 features) et écrit model.pkl à la racine du projet."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    out = ROOT / "model.pkl"
    joblib.dump(model, out)
    print(f"Modèle écrit : {out} (exemple predict [[1.5, 2.5]] -> {model.predict([[1.5, 2.5]])})")


if __name__ == "__main__":
    main()
