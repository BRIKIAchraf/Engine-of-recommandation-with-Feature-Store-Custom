import os
import urllib.request
import zipfile

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    raw_dir = os.path.join("data", "raw")
    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens 1M from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")
        
    extracted_dir = os.path.join(raw_dir, "ml-1m")
    if not os.path.exists(extracted_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_movielens()
