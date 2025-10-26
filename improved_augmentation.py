"""
Improved Graph Augmentation - Torch-Native Implementation
Fixes the numpy conversion bottleneck in the original implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TorchGraphAugmentation(nn.Module):
    """
    Efficient torch-native graph augmentations for DINO training.
    All operations stay on GPU, no numpy conversions.
    """
    
    def __init__(
        self,
        drop_node_prob: float = 0.1,
        drop_edge_prob: float = 0.2,
        mask_feat_prob: float = 0.15,
        add_noise_prob: float = 0.3,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.drop_node_prob = drop_node_prob
        self.drop_edge_prob = drop_edge_prob
        self.mask_feat_prob = mask_feat_prob
        self.add_noise_prob = add_noise_prob
        self.noise_std = noise_std
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        augment_prob: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations with probability.
        
        Args:
            node_features: (N, D) node features
            edge_index: (2, E) edge indices
            augment_prob: Probability to apply each augmentation
        
        Returns:
            augmented_features, augmented_edge_index
        """
        device = node_features.device
        aug_features = node_features.clone()
        aug_edges = edge_index.clone()
        
        # 1. Drop edges (most common)
        if torch.rand(1).item() < augment_prob:
            aug_edges = self.drop_edges(aug_edges)
        
        # 2. Mask features
        if torch.rand(1).item() < augment_prob * 0.7:
            aug_features = self.mask_features(aug_features)
        
        # 3. Add Gaussian noise
        if torch.rand(1).item() < self.add_noise_prob:
            aug_features = self.add_noise(aug_features)
        
        # 4. Drop nodes (less common, more disruptive)
        if torch.rand(1).item() < augment_prob * 0.3:
            aug_features, aug_edges = self.drop_nodes(aug_features, aug_edges)
        
        return aug_features, aug_edges
    
    def drop_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Randomly drop edges."""
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge_prob
        return edge_index[:, keep_mask]
    
    def mask_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """Randomly mask node features (set to 0)."""
        mask = torch.rand_like(node_features) > self.mask_feat_prob
        return node_features * mask
    
    def add_noise(self, node_features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features."""
        noise = torch.randn_like(node_features) * self.noise_std
        return node_features + noise
    
    def drop_nodes(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly drop nodes and update edge indices.
        """
        num_nodes = node_features.size(0)
        keep_mask = torch.rand(num_nodes, device=node_features.device) > self.drop_node_prob
        
        # Keep at least 50% of nodes
        if keep_mask.sum() < num_nodes * 0.5:
            return node_features, edge_index
        
        # New node features
        new_features = node_features[keep_mask]
        
        # Remap edges
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=node_features.device)
        node_map[keep_mask] = torch.arange(keep_mask.sum(), device=node_features.device)
        
        # Filter edges where both endpoints exist
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        new_edges = edge_index[:, edge_mask]
        
        # Remap indices
        new_edges = node_map[new_edges]
        
        return new_features, new_edges


class ConsistentAugmentation(nn.Module):
    """
    Apply the same augmentation to a node's k-hop neighborhood.
    Encourages learning local structure.
    """
    
    def __init__(self, base_augmentor: TorchGraphAugmentation, k_hops: int = 2):
        super().__init__()
        self.augmentor = base_augmentor
        self.k_hops = k_hops
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        anchor_nodes: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply consistent augmentation to local neighborhoods.
        
        Args:
            node_features: (N, D)
            edge_index: (2, E)
            anchor_nodes: Nodes to center augmentation around (default: random)
        """
        if anchor_nodes is None:
            # Select random anchor nodes
            num_anchors = max(1, node_features.size(0) // 100)
            anchor_nodes = torch.randperm(node_features.size(0))[:num_anchors]
        
        # Get k-hop neighborhoods
        neighborhoods = self.get_khop_neighborhoods(edge_index, anchor_nodes, self.k_hops)
        
        # Apply same augmentation to each neighborhood
        aug_features = node_features.clone()
        aug_edges = edge_index.clone()
        
        for neighborhood in neighborhoods:
            # Augment this neighborhood
            hood_mask = torch.zeros(node_features.size(0), dtype=torch.bool, device=node_features.device)
            hood_mask[neighborhood] = True
            
            # Apply masking only to this neighborhood
            if torch.rand(1).item() < 0.5:
                mask = torch.rand(node_features.size(1), device=node_features.device) > self.augmentor.mask_feat_prob
                aug_features[hood_mask] *= mask
        
        return aug_features, aug_edges
    
    @staticmethod
    def get_khop_neighborhoods(edge_index, anchor_nodes, k):
        """BFS to get k-hop neighborhoods."""
        neighborhoods = []
        num_nodes = edge_index.max().item() + 1
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in edge_index.t():
            adj_list[src.item()].append(dst.item())
        
        for anchor in anchor_nodes:
            visited = {anchor.item()}
            current_layer = {anchor.item()}
            
            for _ in range(k):
                next_layer = set()
                for node in current_layer:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            next_layer.add(neighbor)
                            visited.add(neighbor)
                current_layer = next_layer
                if not current_layer:
                    break
            
            neighborhoods.append(torch.tensor(list(visited), dtype=torch.long))
        
        return neighborhoods


def test_augmentation():
    """Test that augmentations work correctly."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample graph
    num_nodes = 100
    node_features = torch.randn(num_nodes, 64, device=device)
    edge_index = torch.randint(0, num_nodes, (2, 300), device=device)
    
    # Test augmentation
    augmentor = TorchGraphAugmentation()
    aug_features, aug_edges = augmentor(node_features, edge_index)
    
    print(f"Original: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"Augmented: {aug_features.size(0)} nodes, {aug_edges.size(1)} edges")
    print(f"Feature change: {(node_features != aug_features).float().mean():.2%}")
    
    # Test consistency
    consistent_aug = ConsistentAugmentation(augmentor, k_hops=2)
    aug_features2, aug_edges2 = consistent_aug(node_features, edge_index)
    print(f"Consistent augmentation applied")


if __name__ == '__main__':
    test_augmentation()