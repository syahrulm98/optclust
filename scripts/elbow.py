import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

def elbow_method(data, title, filename):
    """Determine the optimal number of clusters using the Elbow Method."""
    sse = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.title(f'Elbow Method For Optimal k - {title}')
    plt.savefig(filename)
    plt.close()

    elbow_point = detect_elbow_point(sse, k_range)
    print(f"Optimal number of clusters (Elbow Method): {elbow_point}")
    return elbow_point, sse

def detect_elbow_point(sse, k_range):
    """Detect the elbow point in the SSE plot."""
    deltas = [sse[i] - sse[i + 1] for i in range(len(sse) - 1)]
    elbow_point = k_range[deltas.index(max(deltas)) + 1]
    return elbow_point

def save_results(file_path, results):
    """Save the results to a JSON file."""
    output_file = f"hasil/elbow_optimal_clusters_{file_path.split('/')[-1].split('.')[0]}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Elbow results saved to {output_file}")

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
            
            elbow_k, sse = elbow_method(data, title, f'hasil/elbow_method_{title}.png')
            elbow_results = {
                "file_path": file_path,
                "Elbow Method": {
                    "optimal_k": elbow_k,
                    "sse": sse
                }
            }
            save_results(file_path, elbow_results)

if __name__ == "__main__":
    main()