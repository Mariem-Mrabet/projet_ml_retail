#  Analyse Comportementale Clientèle Retail
### Atelier Machine Learning – GI2 – 2025/2026

---

##  Description du Projet

Ce projet de Machine Learning a pour objectif d’analyser le comportement des clients d’un e-commerce de cadeaux afin de :

-  Segmenter la clientèle
-  Prédire le churn (départ client)
-  Optimiser les stratégies marketing
-  Améliorer la compréhension des profils clients

Le projet suit la chaîne complète Data Science :

Exploration → Préparation → Transformation → Modélisation → Évaluation → Déploiement (Flask)

---

##  Objectifs Techniques

- Nettoyage et préparation de données complexes (52 features)
- Gestion des valeurs manquantes et aberrantes
- Encodage des variables catégorielles
- Normalisation des données
- Réduction de dimension (ACP)
- Clustering client
- Classification (prédiction churn)
- Régression (analyse comportementale)
- Déploiement d’une application web avec Flask

---

##  Structure du Projet

projet_ml_retail/
│
├── data/              # Base de donnees
│   ├── raw/           # Données brutes originales
│   ├── processed/     # Données nettoyées
│   └── train_test/    # Données séparées (train/test)
│
├── notebooks/         # Prototypage et exploration
│
├── src/               # Scripts Python production
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/            # Modèles sauvegardés (.pkl / .joblib)
│
├── app/               # Application web Flask
│
├── reports/           # Rapports et visualisations
│
├── requirements.txt   # Dépendances
├── README.md          # Documentation
└── .gitignore

---

##  Installation

### 1️- Cloner le projet

    git clone https://github.com/Mariem-Mrabet/projet_ml_retail.git
    cd projet_ml_retail

---

### 2️- Créer un environnement virtuel

    python -m venv venv

#Activation sous Windows :

    venv\Scripts\activate

#Activation sous Mac/Linux :

    source venv/bin/activate

---

### 3️- Installer les dépendances

    pip install -r requirements.txt

---

##  Étapes du Projet

### 1️- Exploration des données
- Analyse statistique
- Détection des valeurs manquantes
- Détection des outliers
- Analyse de corrélation

---

### 2️- Préprocessing
- Imputation des valeurs manquantes
- Parsing des dates
- Feature engineering
- Encodage des variables catégorielles
- Normalisation (StandardScaler)
- Suppression des features inutiles

---

### 3️- Séparation Train/Test

train_test_split(test_size=0.2, stratify=y, random_state=42)

---

### 4️- Modélisation

- Classification : Prédiction du churn
- Clustering : Segmentation client
- Régression : Analyse comportementale

Hyperparameter tuning avec :
- GridSearchCV
- (Optionnel) Optuna

---

### 5️- Évaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matrice de confusion

---

### 6️- Déploiement avec Flask

L'application permet :
- Saisie des informations client
- Prédiction du churn
- Affichage du niveau de risque

---

##  Exécution des Scripts

### Préprocessing
    python src/preprocessing.py

### Entraînement du modèle
    python src/train_model.py

### Prédiction
    python src/predict.py

### Lancer l’application Flask
    python app/app.py

Puis ouvrir dans le navigateur :
http://127.0.0.1:5000

---

##  Technologies Utilisées

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- Joblib

---

##  Problèmes de Qualité Traités

- 30% de valeurs manquantes (Age)
- Valeurs aberrantes (SupportTickets, Satisfaction)
- Formats de dates inconsistants
- Variables constantes supprimées
- Déséquilibre des classes (Churn)

---

##  Auteur

Mariem Mrabet  
GI2 – Machine Learning  
Année universitaire 2025–2026  

---

##  Remarque

Ce projet est réalisé dans un cadre pédagogique et suit les bonnes pratiques professionnelles en Data Science et Machine Learning.