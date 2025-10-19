"""
Subgraph sampling strategies for large nucleus graphs
Supports spatial, k-hop, and random walk sampling
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import pickle


class SubgraphSampler:
    """
    Sample subgraphs from large global graphs for mini-batch training.
    Supports multiple sampling strategies for nucleus graphs.
    """
    
    def __init__(
        self,
        strategy: str = 'spatial',  # 'spatial', 'khop', 'random_walk'
        subgraph_size: int = 1000,  # Target number of nodes per subgraph
        overlap: float = 0.2,  # Overlap between spatial patches
        min_nodes: int = 100,  # Minimum nodes to keep a subgraph
        max_nodes: int = 2000,  # Maximum nodes per subgraph
    ):
        """
        Args:
            strategy: Sampling strategy ('spatial', 'khop', 'random_walk')
            subgraph_size: Target number of nodes per subgraph
            overlap: Overlap ratio for spatial sampling (0-0.5)
            min_nodes: Discard subgraphs with fewer nodes
            max_nodes: Maximum nodes per subgraph
        """
        self.strategy = strategy
        self.subgraph_size = subgraph_size
        self.overlap = overlap
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
    def sample_subgraphs(
        self,
        graph_data: Dict,
        num_samples: int = 8,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample multiple subgraphs from a global graph.
        
        Args:
            graph_data: Dictionary with keys:
                - 'node_features': (N, D) array
                - 'edge_index': (E, 2) array  
                - 'centroids': (N, 2) array of spatial coordinates
                - 'edge_attr': (E, F) array (optional)
            num_samples: Number of subgraphs to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of subgraph dictionaries, each with same structure as input
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self.strategy == 'spatial':
            return self._spatial_sampling(graph_data, num_samples)
        elif self.strategy == 'khop':
            return self._khop_sampling(graph_data, num_samples)
        elif self.strategy == 'random_walk':
            return self._random_walk_sampling(graph_data, num_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _spatial_sampling(
        self, 
        graph_data: Dict, 
        num_samples: int
    ) -> List[Dict]:
        """
        Sample subgraphs based on spatial proximity using nucleus centroids.
        Best for nucleus graphs with spatial structure.
        """
        centroids = graph_data['centroids']
        N = len(centroids)
        
        # Estimate patch size to get ~subgraph_size nodes
        node_density = N / (centroids.max(axis=0) - centroids.min(axis=0)).prod()
        patch_area = self.subgraph_size / (node_density + 1e-6)
        patch_size = np.sqrt(patch_area)
        
        # Build KD-tree for efficient spatial queries
        tree = cKDTree(centroids)
        
        subgraphs = []
        attempts = 0
        max_attempts = num_samples * 3
        
        while len(subgraphs) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random center point
            center_idx = np.random.randint(N)
            center = centroids[center_idx]
            
            # Find nodes within patch (with overlap buffer)
            radius = patch_size * (1 + self.overlap)
            node_indices = tree.query_ball_point(center, radius)
            
            # Filter by size constraints
            if len(node_indices) < self.min_nodes:
                continue
            if len(node_indices) > self.max_nodes:
                # Trim to max size by selecting closest nodes
                distances = np.linalg.norm(
                    centroids[node_indices] - center, axis=1
                )
                node_indices = np.array(node_indices)[
                    np.argsort(distances)[:self.max_nodes]
                ].tolist()
            
            # Extract subgraph
            subgraph = self._extract_subgraph(graph_data, node_indices)
            
            # Skip if disconnected or too small
            if len(subgraph['node_features']) >= self.min_nodes:
                subgraphs.append(subgraph)
        
        return subgraphs
    
    def _khop_sampling(
        self,
        graph_data: Dict,
        num_samples: int
    ) -> List[Dict]:
        """
        Sample k-hop neighborhoods around randomly selected seed nodes.
        Good for preserving local graph structure.
        """
        edge_index = graph_data['edge_index']
        N = len(graph_data['node_features'])
        
        # Build adjacency list
        adj_list = [[] for _ in range(N)]
        for src, dst in edge_index:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        subgraphs = []
        attempts = 0
        max_attempts = num_samples * 3
        
        while len(subgraphs) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random seed node
            seed = np.random.randint(N)
            
            # BFS to k-hops
            visited = {seed}
            current_layer = {seed}
            
            for hop in range(10):  # Max hops
                if len(visited) >= self.subgraph_size:
                    break
                    
                next_layer = set()
                for node in current_layer:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            next_layer.add(neighbor)
                            visited.add(neighbor)
                            
                            if len(visited) >= self.max_nodes:
                                break
                    if len(visited) >= self.max_nodes:
                        break
                        
                current_layer = next_layer
                if len(current_layer) == 0:
                    break
            
            node_indices = list(visited)
            
            if len(node_indices) < self.min_nodes:
                continue
            
            subgraph = self._extract_subgraph(graph_data, node_indices)
            if len(subgraph['node_features']) >= self.min_nodes:
                subgraphs.append(subgraph)
        
        return subgraphs
    
    def _random_walk_sampling(
        self,
        graph_data: Dict,
        num_samples: int
    ) -> List[Dict]:
        """
        Sample subgraphs using random walks.
        Creates more cohesive local neighborhoods.
        """
        edge_index = graph_data['edge_index']
        N = len(graph_data['node_features'])
        
        # Build adjacency list
        adj_list = [[] for _ in range(N)]
        for src, dst in edge_index:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        subgraphs = []
        attempts = 0
        max_attempts = num_samples * 3
        
        while len(subgraphs) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Start random walk
            visited = set()
            current = np.random.randint(N)
            
            walk_length = self.subgraph_size * 2  # Over-sample
            for step in range(walk_length):
                visited.add(current)
                
                if len(visited) >= self.max_nodes:
                    break
                
                neighbors = adj_list[current]
                if len(neighbors) == 0:
                    break
                    
                # Random walk step
                current = np.random.choice(neighbors)
            
            node_indices = list(visited)
            
            if len(node_indices) < self.min_nodes:
                continue
            
            subgraph = self._extract_subgraph(graph_data, node_indices)
            if len(subgraph['node_features']) >= self.min_nodes:
                subgraphs.append(subgraph)
        
        return subgraphs
    
    def _extract_subgraph(
        self,
        graph_data: Dict,
        node_indices: List[int]
    ) -> Dict:
        """
        Extract a subgraph given node indices.
        Remaps edges and node features.
        """
        node_indices = np.array(node_indices)
        N_sub = len(node_indices)
        
        # Create node mapping
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Extract node features
        node_features = graph_data['node_features'][node_indices]
        centroids = graph_data['centroids'][node_indices]
        
        # Extract edges within subgraph
        edge_index = graph_data['edge_index']
        edge_mask = np.isin(edge_index[:, 0], node_indices) & \
                    np.isin(edge_index[:, 1], node_indices)
        
        sub_edges = edge_index[edge_mask]
        
        # Remap edge indices
        remapped_edges = np.array([
            [node_map[src], node_map[dst]]
            for src, dst in sub_edges
        ])
        
        # Extract edge attributes if present
        edge_attr = None
        if 'edge_attr' in graph_data and graph_data['edge_attr'] is not None:
            edge_attr = graph_data['edge_attr'][edge_mask]
        
        subgraph = {
            'node_features': node_features,
            'edge_index': remapped_edges,
            'edge_attr': edge_attr,
            'centroids': centroids,
            'num_nodes': N_sub,
            'original_indices': node_indices  # Track original node IDs
        }
        
        return subgraph


class GraphDataLoader:
    """
    DataLoader for large graphs with on-the-fly subgraph sampling.
    Integrates with PyTorch training loops.
    """
    
    def __init__(
        self,
        graph_paths: List[str],
        sampler: SubgraphSampler,
        subgraphs_per_graph: int = 4,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Args:
            graph_paths: List of paths to saved graph pickle files
            sampler: SubgraphSampler instance
            subgraphs_per_graph: Number of subgraphs to sample per global graph
            batch_size: Number of subgraphs per batch
            shuffle: Whether to shuffle graphs
            num_workers: Number of workers (not implemented yet)
        """
        self.graph_paths = graph_paths
        self.sampler = sampler
        self.subgraphs_per_graph = subgraphs_per_graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load all graphs (if memory allows) or keep paths for lazy loading
        self.graphs = []
        self.lazy_load = len(graph_paths) > 100  # Threshold for lazy loading
        
        if not self.lazy_load:
            print(f"Loading {len(graph_paths)} graphs into memory...")
            for path in graph_paths:
                with open(path, 'rb') as f:
                    self.graphs.append(pickle.load(f))
        else:
            print(f"Using lazy loading for {len(graph_paths)} graphs")
            self.graphs = graph_paths  # Keep paths instead
    
    def __len__(self):
        return (len(self.graphs) * self.subgraphs_per_graph) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches of subgraphs."""
        indices = list(range(len(self.graphs)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        batch = []
        
        for idx in indices:
            # Load graph
            if self.lazy_load:
                with open(self.graphs[idx], 'rb') as f:
                    graph = pickle.load(f)
            else:
                graph = self.graphs[idx]
            
            # Sample subgraphs
            subgraphs = self.sampler.sample_subgraphs(
                graph, num_samples=self.subgraphs_per_graph
            )
            
            for subgraph in subgraphs:
                batch.append(subgraph)
                
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []
        
        # Yield remaining
        if len(batch) > 0:
            yield self._collate_batch(batch)
    
    def _collate_batch(self, subgraphs: List[Dict]) -> Dict:
        """
        Collate list of subgraphs into a batched graph.
        Creates batch indices for PyTorch Geometric style batching.
        """
        # Stack node features
        node_features = []
        edge_indices = []
        edge_attrs = []
        batch_indices = []
        
        node_offset = 0
        
        for batch_idx, subgraph in enumerate(subgraphs):
            N = len(subgraph['node_features'])
            
            # Node features
            node_features.append(subgraph['node_features'])
            
            # Edge index (offset by current node count)
            edges = subgraph['edge_index'] + node_offset
            edge_indices.append(edges)
            
            # Edge attributes
            if subgraph['edge_attr'] is not None:
                edge_attrs.append(subgraph['edge_attr'])
            
            # Batch assignment
            batch_indices.extend([batch_idx] * N)
            
            node_offset += N
        
        # Concatenate
        batched = {
            'node_features': np.vstack(node_features),
            'edge_index': np.vstack(edge_indices) if len(edge_indices) > 0 else np.array([]),
            'batch': np.array(batch_indices),
            'num_graphs': len(subgraphs)
        }
        
        if len(edge_attrs) > 0:
            batched['edge_attr'] = np.vstack(edge_attrs)
        
        return batched


# Example usage
if __name__ == '__main__':
    # Load a large graph
    with open('nucleus_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Global graph: {graph['num_nodes']} nodes, "
          f"{len(graph['edge_index'])} edges")
    
    # Create sampler
    sampler = SubgraphSampler(
        strategy='spatial',
        subgraph_size=1000,
        overlap=0.2,
        min_nodes=200,
        max_nodes=2000
    )
    
    # Sample subgraphs
    subgraphs = sampler.sample_subgraphs(graph, num_samples=5)
    
    print(f"\nSampled {len(subgraphs)} subgraphs:")
    for i, sg in enumerate(subgraphs):
        print(f"  Subgraph {i}: {sg['num_nodes']} nodes, "
              f"{len(sg['edge_index'])} edges")
    
    # Create dataloader
    graph_paths = ['graph_1.pkl', 'graph_2.pkl']  # Your graph files
    loader = GraphDataLoader(
        graph_paths=graph_paths,
        sampler=sampler,
        subgraphs_per_graph=4,
        batch_size=8
    )
    
    print(f"\nDataLoader: {len(loader)} batches")
    
    # Iterate
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx}: {len(batch['node_features'])} total nodes, "
              f"{batch['num_graphs']} subgraphs")
        if batch_idx >= 2:
            break