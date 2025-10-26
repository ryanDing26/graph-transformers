"""
Create spatial embedding maps for nucleus graphs.
Similar to tile-based visualizations but for graph subgraphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse
from typing import Dict, List, Tuple
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

import sys
sys.path.append('models')
from student_teacher import StudentTeacherGraphTransformer
from subgraph_sampler import SubgraphSampler


class SpatialEmbeddingMapper:
    """
    Create spatial embedding maps from graph data.
    Two approaches:
    1. Node-level: Embed each nucleus individually
    2. Subgraph-level: Embed spatial patches (like tiles)
    """
    
    def __init__(
        self,
        model: StudentTeacherGraphTransformer,
        device: str = 'cuda',
        use_teacher: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_teacher = use_teacher
    
    def create_subgraph_embedding_map(
        self,
        graph_data: Dict,
        patch_size: float = 1000.0,  # Microns
        overlap: float = 0.5,        # Overlap between patches
        min_nodes: int = 50          # Minimum nodes per patch
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create spatial embedding map using sliding window over tissue.
        This is the direct analog of tile-based approach.
        
        Args:
            graph_data: Full graph dictionary
            patch_size: Size of spatial patches in microns
            overlap: Overlap fraction between patches (0-1)
            min_nodes: Minimum nodes to keep a patch
            
        Returns:
            embeddings: (N_patches, embed_dim) array
            centers: (N_patches, 2) array of patch center coordinates
            node_counts: (N_patches,) array of nodes per patch
        """
        centroids = graph_data['centroids']
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        
        # Get tissue bounds
        x_min, y_min = centroids.min(axis=0)
        x_max, y_max = centroids.max(axis=0)
        
        # Calculate stride
        stride = patch_size * (1 - overlap)
        
        # Generate grid of patch centers
        x_centers = np.arange(x_min, x_max, stride)
        y_centers = np.arange(y_min, y_max, stride)
        
        embeddings = []
        centers = []
        node_counts = []
        
        print(f"Processing {len(x_centers)} x {len(y_centers)} = "
              f"{len(x_centers) * len(y_centers)} patches...")
        
        # Process each patch
        for cx in tqdm(x_centers):
            for cy in y_centers:
                # Find nodes within patch
                distances = np.linalg.norm(centroids - [cx, cy], axis=1)
                patch_mask = distances < (patch_size / 2)
                
                if patch_mask.sum() < min_nodes:
                    continue
                
                # Extract subgraph
                patch_indices = np.where(patch_mask)[0]
                subgraph = self._extract_subgraph(
                    graph_data, 
                    patch_indices
                )
                
                # Get embedding
                embedding = self._embed_subgraph(subgraph)
                
                embeddings.append(embedding)
                centers.append([cx, cy])
                node_counts.append(len(patch_indices))
        
        return (
            np.array(embeddings),
            np.array(centers),
            np.array(node_counts)
        )
    
    def create_node_embedding_map(
        self,
        graph_data: Dict,
        batch_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create embedding for each individual nucleus.
        More granular but slower.
        
        Args:
            graph_data: Full graph dictionary
            batch_size: Process nodes in batches
            
        Returns:
            embeddings: (N_nodes, embed_dim) array
            centroids: (N_nodes, 2) array of nucleus coordinates
        """
        N = len(graph_data['node_features'])
        node_features = graph_data['node_features']
        edge_index = graph_data['edge_index']
        centroids = graph_data['centroids']
        
        # Build adjacency list for k-hop neighborhoods
        adj_list = [[] for _ in range(N)]
        for src, dst in edge_index:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        embeddings = []
        
        print(f"Processing {N} nodes...")
        
        for i in tqdm(range(0, N, batch_size)):
            batch_nodes = list(range(i, min(i + batch_size, N)))
            batch_embeddings = []
            
            for node_idx in batch_nodes:
                # Get k-hop neighborhood
                neighborhood = self._get_khop_neighborhood(
                    node_idx, adj_list, k=3
                )
                
                # Extract subgraph
                subgraph = self._extract_subgraph(
                    graph_data,
                    list(neighborhood)
                )
                
                # Get embedding
                embedding = self._embed_subgraph(subgraph)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings), centroids
    
    @torch.no_grad()
    def _embed_subgraph(self, subgraph: Dict) -> np.ndarray:
        """Get embedding for a subgraph."""
        node_features = torch.from_numpy(subgraph['node_features']).float().to(self.device)
        edge_index = torch.from_numpy(subgraph['edge_index']).long().to(self.device).t()
        
        edge_attr = None
        if subgraph['edge_attr'] is not None:
            edge_attr = torch.from_numpy(subgraph['edge_attr']).float().to(self.device)
        
        if self.use_teacher:
            embedding = self.model.teacher(node_features, edge_index, edge_attr)
        else:
            embedding = self.model.student(node_features, edge_index, edge_attr)
        
        return embedding.cpu().numpy().squeeze()
    
    def _extract_subgraph(self, graph_data: Dict, node_indices: List[int]) -> Dict:
        """Extract subgraph given node indices."""
        node_indices = np.array(node_indices)
        N_sub = len(node_indices)
        
        # Create node mapping
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Extract node features
        node_features = graph_data['node_features'][node_indices]
        
        # Extract edges
        edge_index = graph_data['edge_index']
        edge_mask = np.isin(edge_index[:, 0], node_indices) & \
                    np.isin(edge_index[:, 1], node_indices)
        
        sub_edges = edge_index[edge_mask]
        
        # Remap edges
        remapped_edges = np.array([
            [node_map[src], node_map[dst]]
            for src, dst in sub_edges
        ])
        
        # Extract edge attributes
        edge_attr = None
        if 'edge_attr' in graph_data and graph_data['edge_attr'] is not None:
            edge_attr = graph_data['edge_attr'][edge_mask]
        
        return {
            'node_features': node_features,
            'edge_index': remapped_edges,
            'edge_attr': edge_attr,
            'num_nodes': N_sub
        }
    
    def _get_khop_neighborhood(
        self,
        node_idx: int,
        adj_list: List[List[int]],
        k: int = 3
    ) -> set:
        """Get k-hop neighborhood of a node."""
        visited = {node_idx}
        current_layer = {node_idx}
        
        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        next_layer.add(neighbor)
                        visited.add(neighbor)
            current_layer = next_layer
            if len(current_layer) == 0:
                break
        
        return visited


def visualize_spatial_embeddings(
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    output_path: str,
    method: str = 'cluster',  # 'cluster', 'pca', 'tsne', 'umap'
    n_clusters: int = 10,
    figsize: Tuple[int, int] = (15, 12),
    **kwargs
):
    """
    Visualize spatial embedding map.
    
    Args:
        embeddings: (N, embed_dim) array of embeddings
        coordinates: (N, 2) array of spatial coordinates
        output_path: Where to save the plot
        method: Visualization method
        n_clusters: Number of clusters (for 'cluster' method)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Method 1: Cluster embeddings and color by cluster
    if method == 'cluster':
        print(f"Clustering embeddings into {n_clusters} clusters...")
        
        # Normalize embeddings
        from sklearn.preprocessing import StandardScaler
        embeddings_norm = StandardScaler().fit_transform(embeddings)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_norm)
        
        # Plot
        scatter = axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            cmap='tab20',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[0].set_title('Spatial Embedding Clusters', fontsize=16)
        axes[0].set_xlabel('X (microns)')
        axes[0].set_ylabel('Y (microns)')
        axes[0].axis('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Cluster ID', rotation=270, labelpad=20)
        
        # UMAP of embeddings colored by cluster
        print("Computing UMAP of embeddings...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_norm)
        
        scatter2 = axes[1].scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[1].set_title('UMAP of Embeddings', fontsize=16)
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
    
    # Method 2: PCA on embeddings, color by first PC
    elif method == 'pca':
        print("Computing PCA...")
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(embeddings)
        
        # Spatial map colored by PC1
        scatter = axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=pca_coords[:, 0],
            cmap='RdYlBu_r',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[0].set_title(f'Spatial Map (PC1, {pca.explained_variance_ratio_[0]:.1%} var)', 
                         fontsize=16)
        axes[0].set_xlabel('X (microns)')
        axes[0].set_ylabel('Y (microns)')
        axes[0].axis('equal')
        plt.colorbar(scatter, ax=axes[0], label='PC1 value')
        
        # PCA scatter
        scatter2 = axes[1].scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=pca_coords[:, 2],
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[1].set_title('PCA of Embeddings', fontsize=16)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.colorbar(scatter2, ax=axes[1], label='PC3')
    
    # Method 3: UMAP
    elif method == 'umap':
        print("Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_coords = reducer.fit_transform(embeddings)
        
        # Spatial map colored by UMAP1
        scatter = axes[0].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=umap_coords[:, 0],
            cmap='RdYlBu_r',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[0].set_title('Spatial Map (colored by UMAP1)', fontsize=16)
        axes[0].set_xlabel('X (microns)')
        axes[0].set_ylabel('Y (microns)')
        axes[0].axis('equal')
        plt.colorbar(scatter, ax=axes[0], label='UMAP1')
        
        # UMAP scatter
        scatter2 = axes[1].scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=umap_coords[:, 1],
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='none'
        )
        axes[1].set_title('UMAP of Embeddings', fontsize=16)
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[1], label='UMAP2')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    plt.close()


def visualize_multiple_slides(
    embeddings_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    method: str = 'cluster',
    n_clusters: int = 10
):
    """
    Create spatial embedding maps for multiple slides.
    
    Args:
        embeddings_dict: Dict mapping slide_name -> (embeddings, coordinates)
        output_dir: Directory to save plots
        method: Visualization method
        n_clusters: Number of clusters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for slide_name, (embeddings, coordinates) in embeddings_dict.items():
        print(f"\nProcessing {slide_name}...")
        
        output_path = output_dir / f'{slide_name}_embedding_map.png'
        
        visualize_spatial_embeddings(
            embeddings=embeddings,
            coordinates=coordinates,
            output_path=str(output_path),
            method=method,
            n_clusters=n_clusters
        )


def main():
    parser = argparse.ArgumentParser(
        description='Create spatial embedding maps from nucleus graphs'
    )
    
    # Input
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--graph_file', type=str, required=True,
                       help='Path to nucleus graph .pkl file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for plots')
    
    # Mapping method
    parser.add_argument('--mode', type=str, default='subgraph',
                       choices=['subgraph', 'node'],
                       help='Use subgraph patches (like tiles) or individual nodes')
    
    # Subgraph parameters
    parser.add_argument('--patch_size', type=float, default=1000.0,
                       help='Patch size in microns (for subgraph mode)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap between patches')
    
    # Visualization
    parser.add_argument('--viz_method', type=str, default='cluster',
                       choices=['cluster', 'pca', 'umap'],
                       help='Visualization method')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SPATIAL EMBEDDING MAP GENERATION")
    print("="*80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load graph
    print(f"Loading graph: {args.graph_file}")
    with open(args.graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"  Nodes: {graph_data['num_nodes']}")
    print(f"  Edges: {len(graph_data['edge_index'])}")
    print(f"  Features: {graph_data['node_features'].shape[1]}")
    
    # Reconstruct model
    node_feat_dim = graph_data['node_features'].shape[1]
    model = StudentTeacherGraphTransformer(
        node_feat_dim=node_feat_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        output_dim=1280,
        use_edge_features=False,
        use_teacher_for_inference=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create mapper
    mapper = SpatialEmbeddingMapper(
        model=model,
        device=args.device,
        use_teacher=True
    )
    
    # Generate embedding map
    print(f"\nGenerating {args.mode} embedding map...")
    
    if args.mode == 'subgraph':
        embeddings, coordinates, node_counts = mapper.create_subgraph_embedding_map(
            graph_data=graph_data,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_nodes=50
        )
        
        print(f"  Generated {len(embeddings)} patches")
        print(f"  Nodes per patch: {node_counts.mean():.1f} ± {node_counts.std():.1f}")
        
    else:  # node mode
        embeddings, coordinates = mapper.create_node_embedding_map(
            graph_data=graph_data,
            batch_size=100
        )
        
        print(f"  Generated embeddings for {len(embeddings)} nodes")
    
    # Visualize
    slide_name = Path(args.graph_file).parent.name
    output_path = Path(args.output_dir) / f'{slide_name}_embedding_map.png'
    
    visualize_spatial_embeddings(
        embeddings=embeddings,
        coordinates=coordinates,
        output_path=str(output_path),
        method=args.viz_method,
        n_clusters=args.n_clusters
    )
    
    # Also save raw data
    data_path = Path(args.output_dir) / f'{slide_name}_embedding_data.npz'
    np.savez(
        data_path,
        embeddings=embeddings,
        coordinates=coordinates,
        slide_name=slide_name
    )
    print(f"✓ Saved raw data to {data_path}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()