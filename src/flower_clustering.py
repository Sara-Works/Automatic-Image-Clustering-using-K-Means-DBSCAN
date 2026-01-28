"""
CS412 Project - Simple & Clear Version
Flower Image Clustering using K-Means & DBSCAN
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import time
import pandas as pd

print("=" * 80)
print("CS412 - Flower Image Clustering Project")
print("Simple & Clear Implementation")
print("=" * 80)

# Basic configuration
DATASET_PATH = r"D:\homework\2025-3\AA\AAD_Project\flowers_dataset"
OUTPUT_DIR = "project_results"
SAMPLES = 30  # Images per flower type


# 1. Load images with simple features
def load_images_simple(base_path, samples=30):
    """Load flower images with basic color features"""
    categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    features, labels, paths = [], [], []

    print("\n📥 Loading images...")

    for category in categories:
        path = os.path.join(base_path, category)
        if not os.path.exists(path):
            continue

        images = [f for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.png'))][:samples]

        for img_file in images:
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path)

            if img is not None:
                img_resized = cv2.resize(img, (100, 100))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                # Simple features: average color + simple histogram
                avg_color = np.mean(img_rgb, axis=(0, 1))
                hist_features = []

                for i in range(3):  # R, G, B
                    hist = cv2.calcHist([img_rgb], [i], None, [32], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    hist_features.extend(hist[:8])  # Take first 8 bins

                features.append(list(avg_color) + hist_features)
                labels.append(category)
                paths.append(img_path)

        print(f"  {category}: {len(images)} images")

    print(f"\n Loaded {len(features)} images total")
    return np.array(features), np.array(labels), paths


# 2. Run K-Means
def run_kmeans_simple(X):
    """Simple K-Means implementation"""
    print("\n Running K-Means...")
    start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    time_taken = time.time() - start
    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)

    print(f"  Time: {time_taken:.3f}s")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  DB Index: {db_index:.4f}")

    return {
        'algorithm': 'K-Means',
        'time': time_taken,
        'silhouette': silhouette,
        'db_index': db_index,
        'labels': labels,
        'n_clusters': 5
    }


# 3. Run DBSCAN
def run_dbscan_simple(X):
    """Simple DBSCAN implementation"""
    print("\n🟢 Running DBSCAN...")
    start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    time_taken = time.time() - start

    # Handle noise
    mask = labels != -1
    if sum(mask) > 1:
        silhouette = silhouette_score(X_scaled[mask], labels[mask])
        db_index = davies_bouldin_score(X_scaled[mask], labels[mask])
    else:
        silhouette = -1
        db_index = float('inf')

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"  Time: {time_taken:.3f}s")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")

    return {
        'algorithm': 'DBSCAN',
        'time': time_taken,
        'silhouette': silhouette,
        'db_index': db_index,
        'labels': labels,
        'n_clusters': n_clusters,
        'noise': n_noise
    }


# 4. Create simple visualization
def create_simple_plot(X, true_labels, kmeans_results, dbscan_results):
    """Create basic visualization"""
    # PCA for 2D view
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(15, 4))

    # True distribution
    plt.subplot(1, 3, 1)
    colors = {'daisy': 'red', 'dandelion': 'blue', 'rose': 'green',
              'sunflower': 'orange', 'tulip': 'purple'}

    for flower, color in colors.items():
        mask = true_labels == flower
        if np.any(mask):
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, label=flower, alpha=0.6, s=30)

    plt.title('True Flower Distribution')
    plt.legend(fontsize=8)

    # K-Means results
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=kmeans_results['labels'], cmap='tab10', alpha=0.6, s=30)
    plt.title(f'K-Means Clustering\nTime: {kmeans_results["time"]:.2f}s')

    # DBSCAN results
    plt.subplot(1, 3, 3)
    db_labels = dbscan_results['labels']

    # Plot clusters
    for i in range(dbscan_results['n_clusters']):
        mask = db_labels == i
        if np.any(mask):
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.6, s=30, label=f'Cluster {i + 1}')

    # Plot noise
    noise_mask = db_labels == -1
    if np.any(noise_mask):
        plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                    c='gray', alpha=0.3, s=20, label='Noise')

    plt.title(f'DBSCAN Clustering\nTime: {dbscan_results["time"]:.2f}s')
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('simple_clustering_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: simple_clustering_results.png")
    plt.show()


# 5. Main function
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X, true_labels, image_paths = load_images_simple(DATASET_PATH, SAMPLES)

    if X is None:
        print("Failed to load images")
        return

    # Run algorithms
    kmeans_results = run_kmeans_simple(X)
    dbscan_results = run_dbscan_simple(X)

    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE:")
    print("=" * 80)

    comparison = pd.DataFrame([
        {
            'Algorithm': 'K-Means',
            'Time (s)': f"{kmeans_results['time']:.3f}",
            'Silhouette': f"{kmeans_results['silhouette']:.4f}",
            'DB Index': f"{kmeans_results['db_index']:.4f}",
            'Clusters': kmeans_results['n_clusters'],
            'Noise': 0
        },
        {
            'Algorithm': 'DBSCAN',
            'Time (s)': f"{dbscan_results['time']:.3f}",
            'Silhouette': f"{dbscan_results['silhouette']:.4f}",
            'DB Index': f"{dbscan_results['db_index']:.4f}",
            'Clusters': dbscan_results['n_clusters'],
            'Noise': dbscan_results['noise']
        }
    ])

    print(comparison.to_string(index=False))

    # Save results
    comparison.to_csv(os.path.join(OUTPUT_DIR, 'results_summary.csv'), index=False)

    # Create visualization
    create_simple_plot(X, true_labels, kmeans_results, dbscan_results)

    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
