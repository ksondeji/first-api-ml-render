# first-api-ml-render

Petit projet perso : exposer un modèle de classification via une API REST, puis le déployer en ligne. Première mise en pratique concrète du cycle *modèle → API → cloud*.

---

**Application (prod)** : [Swagger sur Render](https://first-api-ml-render.onrender.com/)


**Documentation interactive (prod)** : [Swagger sur Render](https://first-api-ml-render.onrender.com/docs#/)

---

## Présentation

L’application est une **API FastAPI** qui charge un modèle scikit-learn (`model.pkl`) et expose un endpoint **`POST /predict`**. On envoie deux nombres (`feature1`, `feature2`) en JSON ; l’API renvoie une **classe prédite** (entier). Un **`GET /**` confirme que le service est joignable.

---

## Problématique

En data engineering, un modèle ne sert à rien s’il reste sur un notebook : il faut le **servir** de façon fiable et documentée. 

La question ici : comment passer d’un fichier `.pkl` à un **endpoint HTTP** stable, avec un déploiement **simple** pour l’apprentissage ?

---

## Résultats

- API fonctionnelle en local et sur **Render**, avec doc auto-générée (OpenAPI / Swagger).
- Chaîne claire : entraînement → sérialisation `joblib` → chargement au démarrage → prédiction sur requête.
- Tests de base (`pytest`) sur la racine et `/predict` pour éviter les régressions.

---

## Stack

| Couche | Techno |
|--------|--------|
| API | **FastAPI**, **Pydantic** |
| ML | **scikit-learn**, **joblib** |
| Serveur local | **uvicorn** |
| Hébergement | **Render** |

---

## Installation rapide

```bash
# Cloner le dépôt, puis à la racine du projet :
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Vérifier que **`model.pkl`** est présent à la racine (fourni dans le repo). Sinon, l’app démarre quand même avec un modèle minimal de secours.

```bash
uvicorn main:app --reload
```

- Racine : <http://127.0.0.1:8000/>  
- Doc : <http://127.0.0.1:8000/docs>

**Test rapide** (avec `pytest` installé si besoin : `pip install pytest`) :

```bash
python -m pytest tests/ -v
```

---

