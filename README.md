# first-api-ml-render

Petit projet perso : exposer un modèle de classification via une API REST, puis le déployer en ligne. Première mise en pratique concrète du cycle *modèle → API → cloud*.

---

Application (prod) : [Racine de l’API](https://first-api-ml-render.onrender.com/)

Documentation interactive (prod) : [Swagger / OpenAPI](https://first-api-ml-render.onrender.com/docs)

> Sur la racine `/`, le navigateur affiche du JSON (message de bienvenue) : c’est le comportement attendu. Pour essayer les prédictions, utilise `/docs` ou envoie des requêtes `POST` avec l’en-tête `x-api-key` (valeur actuelle dans le code : `secret`).

---

## Présentation

L’application est une API FastAPI qui charge un modèle scikit-learn (`model.pkl`) au démarrage et expose notamment :

- `POST /predict` : deux nombres (`feature1`, `feature2`) en JSON ; réponse : classe prédite (entier).
- `POST /predict_batch` : liste d’objets du même format ; réponse : liste d’entiers.
- `GET /` : vérifie que le service répond (JSON de bienvenue).

Les routes de prédiction sont protégées par une clé API (`Header` `x-api-key`).

---

## Problématique

Comment passer d’un fichier `.pkl` à un endpoint HTTP stable, avec un déploiement simple pour l’apprentissage ?

---

## Ce qui a été réalisé

- API FastAPI avec schémas Pydantic, logging, garde d’accès par clé API, chargement du modèle au *startup* (`joblib`).
- Modèle : fichier `model.pkl` (régression logistique simple sur deux features), cohérent avec les endpoints
-  script `scripts/train_model.py` pour régénérer ce fichier à partir du même stack (`scikit-learn` + `joblib`).
- Dépendances : `requirements.txt` pour l’installation classique (dont Render) ; `pyproject.toml` pour Poetry (dépendances + groupe *dev* avec tests et couverture).
- Déploiement sur Render (build `pip install -r requirements.txt`, démarrage type `uvicorn main:app --host 0.0.0.0 --port $PORT`).
- Tests automatisés avec pytest et FastAPI `TestClient` : racine, prédiction unitaire et batch, clé API manquante / invalide, données invalides, batch vide.
- Couverture de code : mesure sur le module `main` avec pytest-cov (voir section ci-dessous).

---

## Résultats

- API joignable en local et sur Render, documentation OpenAPI / Swagger générée automatiquement.
- Chaîne claire : entraînement → sérialisation `joblib` → chargement au démarrage → prédiction à la requête.
- Suite de tests qui limite les régressions sur les parcours principaux et la sécurité basique par clé.

---

## Tests

Les tests vivent dans le dossier `tests/` (fichier `test_api.py`). Ils utilisent un mock du modèle ML pour ne pas dépendre du chargement réel de `model.pkl` pendant les tests.

Avec pip (après `pip install -r requirements.txt`, et en local `pip install pytest-cov` si tu veux la couverture) :

```bash
cd chemin/vers/first-api-ml-render
python -m pytest tests/ -v
```

Avec Poetry :

```bash
poetry install
poetry run pytest -v
```

Le fichier `pyproject.toml` configure `testpaths = ["tests"]` et `pythonpath = ["."]` pour que le fichier `main.py` fonctionne depuis les tests.

---

## Couverture de code

La couverture est calculée sur `main.py` (module `main`), avec branche activée (`branch = true` dans `pyproject.toml`).

Dernière mesure de référence : environ 77 % de couverture sur `main.py` (7 tests passants). Les zones non couvertes correspondent surtout à :

- le `try` / `except` du chargement du modèle au démarrage (`load_model`) : en tests, le *startup* réel n’est pas rejoué avec un échec de `joblib.load` ;
- les blocs `except` des routes `/predict` et `/predict_batch` : les tests ne déclenchent pas d’erreur interne sur `model.predict`, donc les chemins d’erreur HTTP 500 ne sont pas exécutés.

Pour afficher le détail dans le terminal :

```bash
poetry run pytest --cov=main --cov-report=term-missing
```

Rapport HTML (dossier `htmlcov/`) :

```bash
poetry run pytest --cov=main --cov-report=html
```

*(Avec pip : `python -m pytest tests/ --cov=main --cov-report=term-missing` après `pip install pytest-cov`.)*

---

## Stack

| Couche | Techno |
|--------|--------|
| API | FastAPI, Pydantic |
| ML | scikit-learn, joblib |
| Serveur local | uvicorn |
| Tests | pytest, httpx (client de test), pytest-cov (couverture) |
| Gestion deps (local) | Poetry (`pyproject.toml`) |
| Hébergement | Render |

---

## Installation rapide

```bash
# Cloner le dépôt, puis à la racine du projet :
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Le fichier `model.pkl` doit être présent à la racine du dépôt : l’application charge ce fichier au démarrage ; en cas d’échec du chargement, le service ne démarre pas (pas de modèle de secours automatique).

Régénérer le modèle (optionnel) :

```bash
python scripts/train_model.py
```

Lancer l’API en local :

```bash
uvicorn main:app --reload
```

- Racine : <http://127.0.0.1:8000/>
- Doc : <http://127.0.0.1:8000/docs>

---

## Installation avec Poetry (optionnel)

```bash
poetry install
poetry run uvicorn main:app --reload
```

---

## Rappel Render

- Build : `pip install -r requirements.txt`
- Start : `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Pense à versionner `model.pkl` si Render clone le dépôt sans autre source d’artefacts.

---
