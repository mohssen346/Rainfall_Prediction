# src/02_clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_rainfall(input_path: str, output_path: str, n_clusters: int = 5, max_rainfall: float = 500):
    """
    Perform KMeans clustering on daily rainfall (rrr24) and assign ordered labels.
    """
    df = pd.read_csv(input_path)
    
    # Remove extreme outliers
    df = df[df['rrr24'] <= max_rainfall].copy()
    
    if 'rrr24' not in df.columns:
        raise ValueError("Column 'rrr24' not found in the dataset!")

    # Standardize rainfall
    scaler = StandardScaler()
    df['rrr24_scaled'] = scaler.fit_transform(df[['rrr24']])

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['rrr24_scaled']])

    # Order clusters by increasing rainfall intensity
    cluster_ranges = df.groupby('Cluster')['rrr24'].agg(['min', 'max']).sort_values('min')
    print("Rainfall range for each cluster:")
    print(cluster_ranges)

    # Remap cluster labels: 0 = lowest rainfall, 4 = highest
    order_map = {old: new for new, old in enumerate(cluster_ranges.index)}
    df['Cluster'] = df['Cluster'].map(order_map)

    df.drop(columns=['rrr24_scaled'], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Clustered data saved to {output_path}")

if __name__ == "__main__":
    cluster_rainfall(
        input_path="data/combined_output.csv",
        output_path="data/clustered_data.csv",
        n_clusters=5
    )