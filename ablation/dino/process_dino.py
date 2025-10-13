import h5py
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["v2", "v3"], help="DiNO model to use: 'v2' or 'v3'")
    args = parser.parse_args()

    with h5py.File(f"./embeddings/embeddings_with_coordinates_{args.model_type}.h5", "r") as f:
        for slide_name in f.keys():
            print(f[slide_name].keys())
            slide_emb = f[slide_name]["slide_embedding"][:]
            patch_embs = f[slide_name]["patch_embeddings"][:]
            coords = f[slide_name]["coordinates"][:]
            patch_embs = StandardScaler().fit_transform(patch_embs)
            kmeans = KMeans(n_clusters=5, random_state=42)
            labels = kmeans.fit_predict(patch_embs)
            # Assign each cluster label a color
            palette = sns.color_palette("tab10", n_colors=10)
            plt.figure(figsize=(8, 8))
            for i in range(len(labels)):
                plt.scatter(coords[i][0], coords[i][1], s=10, color=palette[labels[i]], label=labels[i], alpha=0.7)
            plt.axis("equal")
            plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f"Clusters for Sample {slide_name}")
            plt.savefig(f"./plots/dino{args.model_type}/{slide_name}.png", dpi=300)

if __name__ == "__main__":
    main()