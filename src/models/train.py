import os
import pandas as pd
import lightgbm as lgb
from surprise import SVD, Dataset, Reader
from surprise.model_selection import PredefinedFold
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
from feast import FeatureStore
from sklearn.metrics import mean_squared_error
import numpy as np

def prepare_entity_df(ratings_path):
    print("Preparing entity dataframe...")
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings['event_timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    # Sort by timestamp to prevent data leakage in train/test split
    ratings = ratings.sort_values('event_timestamp')
    return ratings

def train_svd(train_df, test_df):
    print("Training SVD Baseline...")
    reader = Reader(rating_scale=(1, 5))
    
    trainset = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader).build_full_trainset()
    testset = list(test_df[['user_id', 'movie_id', 'rating']].itertuples(index=False, name=None))
    
    algo = SVD(random_state=42)
    algo.fit(trainset)
    
    predictions = algo.test(testset)
    rmse = np.sqrt(np.mean([(pred.est - pred.r_ui)**2 for pred in predictions]))
    return algo, rmse

def train_lgb(train_df, test_df):
    print("Training LightGBM...")
    # Drop columns that are not features
    drop_cols = ['user_id', 'movie_id', 'rating', 'timestamp', 'event_timestamp', 'gender', 'zip', 'title', 'genres']
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['rating']
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['rating']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    gbm = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
    
    preds = gbm.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return gbm, rmse

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Recommendation_Engine")
    
    store = FeatureStore(repo_path="feature_repo")
    
    ratings_path = "data/raw/ml-1m/ratings.dat"
    entity_df = prepare_entity_df(ratings_path)
    
    # 80/20 sequential split
    split_idx = int(len(entity_df) * 0.8)
    train_entity_df = entity_df.iloc[:split_idx]
    test_entity_df = entity_df.iloc[split_idx:]
    
    features = [
        "user_features:age",
        "user_features:occupation",
        "user_features:user_avg_rating",
        "user_features:user_nb_movies",
        "user_features:user_rating_stddev",
        "movie_features:release_year",
        "movie_features:movie_avg_rating",
        "movie_features:movie_total_views",
        "movie_features:movie_rating_stddev",
    ]
    
    print("Fetching historical features for training...")
    train_df = store.get_historical_features(
        entity_df=train_entity_df,
        features=features
    ).to_df()
    
    print("Fetching historical features for testing...")
    test_df = store.get_historical_features(
        entity_df=test_entity_df,
        features=features
    ).to_df()
    
    # Fill NA for missing features
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    with mlflow.start_run(run_name="Baseline_SVD"):
        svd_model, svd_rmse = train_svd(train_df, test_df)
        mlflow.log_metric("rmse", svd_rmse)
        print(f"SVD RMSE: {svd_rmse}")
        # Log model could be done with a custom flavor
        
    with mlflow.start_run(run_name="LightGBM_Feast"):
        lgb_model, lgb_rmse = train_lgb(train_df, test_df)
        mlflow.log_metric("rmse", lgb_rmse)
        mlflow.lightgbm.log_model(lgb_model, "model")
        print(f"LightGBM RMSE: {lgb_rmse}")
        
if __name__ == "__main__":
    main()
