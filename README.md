# 🎬 Moteur de Recommandation Hybride avec Feature Store Personnalisé & MLOps

[![Feast Version](https://img.shields.io/badge/Feast-0.38.0-blue.svg?style=for-the-badge&logo=fastapi)](https://github.com/feast-dev/feast)
[![Spark Version](https://img.shields.io/badge/PySpark-3.5.1-orange.svg?style=for-the-badge&logo=apachespark)](https://spark.apache.org/)
[![MLflow Version](https://img.shields.io/badge/MLflow-Tracking-green.svg?style=for-the-badge&logo=mlflow)](https://mlflow.org/)
[![FastAPI Version](https://img.shields.io/badge/FastAPI-Serving-red.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker Compose](https://img.shields.io/badge/Docker_Compose-Infrastructure-blue.svg?style=for-the-badge&logo=docker)](https://www.docker.com/)

Ce projet présente un **système de recommandation hybride de niveau production** utilisant un **Feature Store personnalisé (Feast)**, des pipelines de calcul distribués (**PySpark**), un suivi rigoureux des expériences (**MLflow**), une orchestration complète (**Apache Airflow**), une API de service à faible latence (**FastAPI**), et un système automatisé de détection de dérive des données (**Data Drift Monitoring via PSI**). 

L'ensemble de la pile est conteneurisé à l'aide de **Docker & Docker Compose** pour simplifier le déploiement et garantir la reproductibilité. Le jeu de données utilisé pour ce projet est le célèbre **MovieLens 1M**.

---

## 📐 Architecture Globale du Système

L'architecture repose sur la séparation nette entre le **traitement hors ligne (Offline)**, le **stockage et le service en ligne (Online Features)**, l'**entraînement et le suivi des modèles**, et le **service de prédiction en temps réel**.

### Diagramme d'Architecture

```mermaid
flowchart TB
    %% Nodes definition
    subgraph Data_Layer ["1. Ingestion & Feature Engineering"]
        ML1M[("MovieLens 1M Raw Data<br>(ratings, users, movies)")]
        SparkJob["⚡ PySpark Processor<br>(spark_features.py)"]
        ParquetStore[("📁 Parquet Feature Store<br>(Partitioned by event_timestamp)")]
    end

    subgraph Feature_Store_Layer ["2. Feature Store Architecture (Feast)"]
        FeastApply["🛠️ Feast Registry<br>(PostgreSQL Store)"]
        RedisOnline[("🎯 Redis Online Store<br>(Fast Feature Serving)")]
    end

    subgraph Training_Layer ["3. Entraînement & Expérimentation"]
        TrainScript["🏋️ Train Pipeline<br>(train.py)"]
        MLflowServer["📊 MLflow Tracking Server<br>(Postgres Metadata)"]
        SVDModel["🔵 Surprise SVD Model<br>(Baseline)"]
        LGBMModel["🟢 LightGBM Model<br>(Feast Trained)"]
    end

    subgraph Serving_Layer ["4. Serving en Temps Réel (API)"]
        FastAPI_App["🚀 FastAPI Recommendation Service<br>(main.py)"]
        UserRequest["👤 Utilisateur / Application Client"]
    end

    subgraph Monitoring_Layer ["5. Observabilité & Drift"]
        DriftScript["🚨 Drift Detector (PSI)<br>(drift_detection.py)"]
    end

    %% Data Flow
    ML1M -->|Fichiers Plats .dat| SparkJob
    SparkJob -->|Aggrégations Distribuées| ParquetStore
    ParquetStore -->|Enregistrement dans schema| FeastApply
    FeastApply -->|Matérialisation Incremental| RedisOnline

    %% Training Flow
    ParquetStore -->|get_historical_features| TrainScript
    TrainScript -->|Log Expériences & Métriques| MLflowServer
    TrainScript -->|Modèle A| SVDModel
    TrainScript -->|Modèle B| LGBMModel

    %% Serving Flow
    UserRequest -->|GET /recommend/{user_id}| FastAPI_App
    FastAPI_App -->|A/B Testing Router| Router{Route Strategy?}
    Router -->|Option A: SVD| SVDModel
    Router -->|Option B: LightGBM| RedisOnline
    RedisOnline -->|get_online_features (User & Movie)| FastAPI_App
    LGBMModel -->|Inférence en ligne| FastAPI_App
    FastAPI_App -->|JSON Recommendations & Latency| UserRequest

    %% Monitoring Flow
    ParquetStore -->|Distribution Baseline| DriftScript
    RedisOnline -->|Distribution Active| DriftScript
    DriftScript -->|Alerte Drift > 0.2| Slack_Console["🚨 Console / Alerting"]
```

---

## 🛠️ Pile Technologique

*   **Calcul & Feature Engineering :** **PySpark (v3.5.1)** pour le calcul distribué à grande échelle des statistiques utilisateurs et films.
*   **Feature Store :** **Feast (v0.38.0)**
    *   *Registre Hors ligne :* Base de données **PostgreSQL** pour stocker les métadonnées et schémas Feast.
    *   *Stockage Hors ligne :* Fichiers **Parquet** locaux (partitionnés).
    *   *Stockage En ligne :* **Redis** pour la récupération à très faible latence (< 10ms) des caractéristiques (features) en ligne.
*   **Modélisation & Machine Learning :**
    *   **Scikit-Surprise :** Entraînement d'un modèle de décomposition en valeurs singulières (**SVD**) comme ligne de base (Baseline).
    *   **LightGBM :** Modèle de classement basé sur les arbres de décision entraîné sur les caractéristiques Feast (historique des notes, âges, professions, moyennes des films).
*   **Suivi & MLOps :** **MLflow** avec un serveur centralisé stockant les paramètres, métriques (RMSE) et artefacts de modèles sur Postgres et le système de fichiers local.
*   **Orchestration :** **Apache Airflow (LocalExecutor)** pour planifier et orchestrer le pipeline de données (téléchargement, calcul PySpark, Feast materialize).
*   **Service API :** **FastAPI** et **Uvicorn** proposant un routage d'A/B Testing en temps réel avec mesure de latence.
*   **Surveillance :** Script personnalisé de calcul de l'**Indice de Stabilité de la Population (PSI)** pour le monitoring de dérive des données.
*   **Déploiement :** **Docker & Docker Compose** configurant 7 services interconnectés.

---

## 📂 Structure du Projet

```text
├── feature_repo/                      # Configuration et définitions du Feature Store Feast
│   ├── feature_store.yaml             # Paramètres Feast (Redis, PostgreSQL registry, providers)
│   └── features.py                    # Définitions des entités, sources et Feature Views
├── notebooks/
│   └── 01_exploration.ipynb           # Notebook d'exploration de données et prototypes
├── scripts/
│   ├── download_data.py               # Télécharge et extrait MovieLens 1M
│   └── feast_apply_materialize.sh     # Applique Feast et matérialise les features vers Redis
├── src/
│   ├── api/
│   │   └── main.py                    # API FastAPI avec A/B testing et intégration Feast Online
│   ├── features/
│   │   └── spark_features.py          # Script de calcul des features à l'aide de PySpark
│   ├── models/
│   │   └── train.py                   # Entraînement des modèles SVD & LightGBM et suivi MLflow
│   └── monitoring/
│       └── drift_detection.py         # Calcul du PSI et alerte de dérive des caractéristiques
├── Dockerfile.airflow                 # Dockerfile pour le service Apache Airflow
├── Dockerfile.jupyter                 # Dockerfile pour Jupyter et MLflow
├── docker-compose.yml                 # Orchestration de tous les services Docker
├── docker-init/
│   └── init.sql                       # Initialisation des bases PostgreSQL (Feast, Airflow, MLflow)
└── requirements.txt                   # Dépendances Python du projet
```

---

## ⚙️ Guide de Démarrage Rapide

### 1. Prérequis
Assurez-vous d'avoir installé :
*   Docker et Docker Compose.
*   Un client Git.

### 2. Démarrage des Services de l'Infrastructure
Lancez l'ensemble des conteneurs en arrière-plan :

```bash
docker-compose up -d --build
```

Cette commande démarre :
1.  **PostgreSQL** (Port `5432`) : Héberge les bases de données d'Airflow, de Feast et de MLflow.
2.  **Redis** (Port `6379`) : Magasin de fonctionnalités en ligne Feast.
3.  **JupyterLab** (Port `8888`) : Environnement de développement et d'exploration.
4.  **MLflow Server** (Port `5000`) : Serveur de suivi des modèles.
5.  **Airflow Webserver & Scheduler** (Port `8080`) : Interface d'orchestration.

---

## 🚀 Exécution pas-à-pas du Pipeline MLOps

Pour exécuter les étapes à l'intérieur du conteneur de développement (Jupyter/MLflow), vous pouvez ouvrir une session interactive ou exécuter directement les commandes suivantes.

### Étape 1 : Téléchargement des Données
Téléchargez le jeu de données MovieLens 1M :

```bash
docker-compose exec jupyter python scripts/download_data.py
```

### Étape 2 : Calcul des Features avec PySpark
Générez les features aggrégées pour les utilisateurs et les films, puis sauvegardez-les au format Parquet :

```bash
docker-compose exec jupyter python src/features/spark_features.py
```

*   **Caractéristiques Utilisateur calculées :** `gender`, `age`, `occupation`, `zip`, `user_avg_rating` (note moyenne attribuée par l'utilisateur), `user_nb_movies` (nombre de films notés), `user_rating_stddev` (écart-type des notes).
*   **Caractéristiques Film calculées :** `title`, `genres`, `release_year` (année extraite du titre), `movie_avg_rating` (note moyenne du film), `movie_total_views` (nombre total de vues/notes), `movie_rating_stddev` (écart-type des notes).

### Étape 3 : Application & Matérialisation du Feature Store Feast
Enregistrez les schémas dans PostgreSQL et chargez les caractéristiques dans la base Redis en ligne :

```bash
docker-compose exec jupyter bash scripts/feast_apply_materialize.sh
```

### Étape 4 : Entraînement des Modèles & Suivi MLflow
Entraînez le modèle Baseline (SVD) et le modèle de classement refined (LightGBM) en extrayant les données historiques via Feast :

```bash
docker-compose exec jupyter python src/models/train.py
```

*   Le script effectue une **séparation séquentielle (chronologique)** à 80/20 pour éviter la fuite de données (*data leakage*).
*   Il appelle `store.get_historical_features` pour joindre dynamiquement les Parquet de features aux notes d'entraînement.
*   Il enregistre les modèles et les métriques RMSE sur MLflow. Vous pouvez consulter l'interface sur `http://localhost:5000`.

---

## 📡 Service de Recommandation en Temps Réel (FastAPI)

Le service FastAPI (`src/api/main.py`) propose un point de terminaison de recommandation performant.

### Démarrer le serveur API localement ou via Docker
Le serveur API est configuré pour écouter les requêtes de recommandation. Lancez-le dans le conteneur jupyter :

```bash
docker-compose exec -d jupyter uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Stratégie d'A/B Testing en Temps Réel
L'API prend en charge le routage dynamique basé sur l'en-tête de requête HTTP `x-strategy` :

1.  **Stratégie A (SVD Matrix Factorization) :** Retourne des scores rapides et personnalisés basés sur les interactions utilisateur-film collaboratives globales.
2.  **Stratégie B (LightGBM + Feast Online Store) :**
    *   Récupère en temps réel les fonctionnalités de l'utilisateur concerné depuis **Redis** via Feast.
    *   Récupère les fonctionnalités des 100 films candidats à recommander.
    *   Combine et diffuse les caractéristiques dans un modèle de régression LightGBM pour classer les films candidats par pertinence.

Si aucun en-tête `x-strategy` n'est fourni, l'API distribue les requêtes de manière égale (**50/50**) entre les deux stratégies pour tester la performance en conditions réelles.

#### Exemple de requête (Stratégie B forcée) :
```bash
curl -H "x-strategy: B" http://localhost:8000/recommend/10?n=5
```

#### Exemple de réponse :
```json
{
  "user_id": 10,
  "strategy": "B",
  "latency_ms": 14.82,
  "recommendations": [
    { "movie_id": 42, "score": 4.87 },
    { "movie_id": 18, "score": 4.62 },
    { "movie_id": 91, "score": 4.51 },
    { "movie_id": 5,  "score": 4.39 },
    { "movie_id": 73, "score": 4.15 }
  ]
```

---

## 🚨 Détection de Dérive de Données (Drift Detection - PSI)

Pour s'assurer que les modèles de recommandation ne souffrent pas de dégradation de performance due à la dérive des données (Data Drift), un système de surveillance basé sur l'**Indice de Stabilité de la Population (PSI)** est implémenté (`src/monitoring/drift_detection.py`).

### Fondement Mathématique du PSI

Le PSI mesure à quel point la distribution d'une variable cible (par exemple, les notes attribuées par les utilisateurs) a changé entre deux périodes ou jeux de données (ici, la distribution historique d'entraînement vs la distribution active en ligne récupérée de Redis).

La formule mathématique est la suivante :

$$PSI = \sum_{i=1}^{k} \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$

Où :
*   $P_i$ représente la proportion d'observations dans l'intervalle (bucket) $i$ pour la population **actuelle (online)**.
*   $Q_i$ représente la proportion d'observations dans l'intervalle $i$ pour la population de **référence (offline/entraînement)**.
*   $k$ est le nombre total d'intervalles (ici réglé sur 10 déciles).

### Seuils d'évaluation du PSI :
*   **$PSI \le 0.1$** : ✅ **Pas de dérive significative**. Les distributions sont stables.
*   **$0.1 < PSI \le 0.2$** : ⚠️ **Dérive mineure**. Une surveillance est requise, les caractéristiques commencent à changer.
*   **$PSI > 0.2$** : 🚨 **Dérive critique**. Une alerte est déclenchée. Il est recommandé de ré-entraîner le modèle sur des données plus fraîches.

### Exécuter le diagnostic de Drift :
Pour calculer le PSI sur les distributions de `user_avg_rating` en comparant la vérité terrain (Parquet historique) avec la production en direct (Redis) :

```bash
docker-compose exec jupyter python src/monitoring/drift_detection.py
```

---

## 📈 Suivi des Modèles dans MLflow

Le serveur MLflow centralise vos expériences. Ouvrez votre navigateur sur `http://localhost:5000`.

Vous y trouverez :
*   **La comparaison des exécutions (runs) :** Comparez facilement les métriques de RMSE entre votre Baseline SVD et votre modèle LightGBM.
*   **L'enregistrement des artefacts :** MLflow stocke la structure de votre modèle LightGBM entraîné pour faciliter sa mise en production future.
*   **La traçabilité complète :** Les hyperparamètres du modèle de gradient boosting (learning rate, leaves...) sont loggués automatiquement.

---

## 📌 Orchestration des Tâches avec Apache Airflow

L'ensemble du pipeline peut être automatisé et exécuté périodiquement via l'interface Apache Airflow accessible sur `http://localhost:8080` (identifiants: `admin` / `admin`).

Les DAGs configurés orchestreront périodiquement :
1.  Le rapatriement ou la mise à jour des notes de films.
2.  L'exécution de la tâche Spark de calcul de features.
3.  La matérialisation incrémentale Feast (`feast materialize-incremental`) pour rafraîchir le magasin en ligne Redis sans recharger tout l'historique.
4.  L'évaluation continue de la dérive (PSI).
