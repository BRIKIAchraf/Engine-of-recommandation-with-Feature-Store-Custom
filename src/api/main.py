from fastapi import FastAPI, Header, HTTPException
from typing import Optional
from feast import FeatureStore
import pandas as pd
import random
import time
import mlflow.pyfunc
import numpy as np

app = FastAPI(title="Recommendation API")

# Initialize Feast
store = FeatureStore(repo_path="feature_repo")

# We will load models lazily or globally. In a real app, do it on startup.
# lgb_model = mlflow.pyfunc.load_model("models:/LightGBM_Model/Production")
# svd_model: we would need a custom way to load the surprise SVD model, or just use a dummy for the demo.

candidate_movies = list(range(1, 101)) # Simulating a candidate generation step (e.g. top 100 movies)

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10, x_strategy: Optional[str] = Header(None)):
    start_time = time.time()
    
    # Random 50/50 routing if header is not provided
    if not x_strategy:
        x_strategy = random.choice(["A", "B"])
        
    x_strategy = x_strategy.upper()
    
    recommendations = []
    
    if x_strategy == "A":
        # Strategy A: Pure SVD (Simulated here if model is not loaded)
        # In a real scenario, we'd iterate over candidate_movies and predict with svd_model.predict(user_id, m)
        # For demo purposes, returning random scores
        scores = [(m, random.uniform(2.5, 5.0)) for m in candidate_movies]
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [{"movie_id": m, "score": s} for m, s in scores[:n]]
        
    elif x_strategy == "B":
        # Strategy B: LightGBM + Feast
        # 1. Fetch user features
        user_features_dict = store.get_online_features(
            features=[
                "user_features:age",
                "user_features:occupation",
                "user_features:user_avg_rating",
                "user_features:user_nb_movies",
                "user_features:user_rating_stddev",
            ],
            entity_rows=[{"user_id": user_id}]
        ).to_dict()
        
        # 2. Fetch movie features for candidates
        movie_features_dict = store.get_online_features(
            features=[
                "movie_features:release_year",
                "movie_features:movie_avg_rating",
                "movie_features:movie_total_views",
                "movie_features:movie_rating_stddev",
            ],
            entity_rows=[{"movie_id": m} for m in candidate_movies]
        ).to_df()
        
        # 3. Construct DataFrame for LightGBM
        # broadcast user features
        user_df = pd.DataFrame(user_features_dict)
        # For simplicity, assign a dummy score
        # In reality: preds = lgb_model.predict(combined_df)
        scores = [(row['movie_id'], random.uniform(3.0, 5.0)) for _, row in movie_features_dict.iterrows()]
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [{"movie_id": m, "score": s} for m, s in scores[:n]]
    else:
        raise HTTPException(status_code=400, detail="Strategy must be A or B")
        
    latency_ms = (time.time() - start_time) * 1000
    
    # Log prediction details to console/logs (could be sent to a tracking DB or Kafka)
    print(f"User: {user_id}, Strategy: {x_strategy}, Latency: {latency_ms:.2f}ms")
    
    return {
        "user_id": user_id,
        "strategy": x_strategy,
        "latency_ms": latency_ms,
        "recommendations": recommendations
    }
