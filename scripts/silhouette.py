import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

def load_data(file_path):
    """Load data from a CSV or Excel file."""
    try:
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path, delimiter=';')
        print(f"Data from {file_path} loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_data(data):
    """Preprocess data by handling missing values and converting categorical data."""
    data = data.dropna()
    for column in data.columns:
        if data[column].dtype == 'object':
            try:
                data[column] = data[column].str.replace(';', '.').astype(float)
            except ValueError:
                data[column] = pd.factorize(data[column])[0]
    return data

def silhouette_method(data, title, filename):
    """Determine the optimal number of clusters using the Silhouette Method."""
    silhouette_avg = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg.append(silhouette_score(data, cluster_labels))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_avg, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Method For Optimal k - {title}')
    plt.savefig(filename)
    plt.close()

    optimal_k = k_range[silhouette_avg.index(max(silhouette_avg))]
    print(f"Optimal number of clusters (Silhouette Method): {optimal_k}")
    return optimal_k, silhouette_avg

def save_results(file_path, results):
    """Save the results to a JSON file."""
    output_file = f"hasil/silhouette_optimal_clusters_{file_path.split('/')[-1].split('.')[0]}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Silhouette results saved to {output_file}")

def main():
    file_paths = [
        'data/Data-Preferensi.xlsx',
        'data/Data-Studi-Kasus.xlsx',
        'data/Data-Preferensi.csv',
        'data/Data-Studi-Kasus.csv'
    ]

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        data = load_data(file_path)
        if data is not None:
            data = preprocess_data(data)
            title = file_path.split('/')[-1].split('.')[0]
            
            silhouette_k, silhouette_scores = silhouette_method(data, title, f'hasil/silhouette_method_{title}.png')
            silhouette_results = {
                "file_path": file_path,
                "Silhouette Method": {
                    "optimal_k": silhouette_k,
                    "silhouette_scores": silhouette_scores
                }
            }
            save_results(file_path, silhouette_results)

if __name__ == "__main__":
    main()