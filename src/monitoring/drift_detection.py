import numpy as np
import pandas as pd
from feast import FeatureStore

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate the PSI (Population Stability Index) across all variables
    """
    def psi_single(expected_array, actual_array, buckets):
        def scale_range(input_array, min_val, max_val):
            input_array += -(np.min(input_array))
            input_array /= np.max(input_array) / (max_val - min_val)
            input_array += min_val
            return input_array

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        if buckettype == 'bins':
            breakpoints = np.percentile(expected_array, breakpoints)
            
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        
        # Replace 0 to avoid div by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        value = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        return np.sum(value)

    if len(expected.shape) == 1:
        psi_values = psi_single(expected, actual, buckets)
    else:
        psi_values = []
        for i in range(expected.shape[1]):
            psi_values.append(psi_single(expected[:, i], actual[:, i], buckets))
            
    return psi_values

def main():
    print("Démarrage de la détection de Drift (PSI)...")
    
    store = FeatureStore(repo_path="feature_repo")
    
    # Simulating training data distribution (normally loaded from training dataset artifacts)
    # We load offline features to represent 'expected'
    print("Chargement des features offline (training/expected)...")
    # Using a dummy entity DataFrame for offline fetch
    entity_df = pd.DataFrame({'user_id': range(1, 1001)})
    entity_df['event_timestamp'] = pd.to_datetime('today')
    
    try:
        training_data = store.get_historical_features(
            entity_df=entity_df,
            features=["user_features:user_avg_rating"]
        ).to_df()
        expected = training_data['user_avg_rating'].dropna().values
    except Exception as e:
        print(f"Erreur lors de la récupération offline (peut-être non matérialisé) : {e}")
        # Dummy data
        expected = np.random.normal(loc=3.5, scale=1.0, size=1000)

    # Fetching online data from Redis (actual)
    print("Chargement des features online depuis Redis (actual)...")
    try:
        online_data = store.get_online_features(
            features=["user_features:user_avg_rating"],
            entity_rows=[{"user_id": u} for u in range(1, 1001)]
        ).to_df()
        actual = online_data['user_avg_rating'].dropna().values
    except Exception as e:
        print(f"Erreur lors de la récupération online : {e}")
        # Dummy data (slightly shifted to simulate drift)
        actual = np.random.normal(loc=3.8, scale=1.1, size=1000)

    if len(expected) > 0 and len(actual) > 0:
        psi_value = calculate_psi(expected, actual)
        print(f"Index de Stabilité de Population (PSI) : {psi_value:.4f}")
        
        if psi_value > 0.2:
            print("🚨 ALERTE : Dérive des données détectée (Drift > 0.2) !")
        elif psi_value > 0.1:
            print("⚠️ AVERTISSEMENT : Dérive mineure détectée (0.1 < Drift <= 0.2).")
        else:
            print("✅ OK : Pas de dérive significative (Drift <= 0.1).")
    else:
        print("Données insuffisantes pour calculer le PSI.")

if __name__ == "__main__":
    main()
