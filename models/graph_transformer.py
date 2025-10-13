import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer with multi-head attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_edge_features: bool = False
    ):
        super().__init__()
        
        self.attention = GraphMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_edge_features=use_edge_features
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, adj_mask, edge_index, edge_attr)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class GraphMultiHeadAttention(nn.Module):
    """Multi-head attention for graphs with optional edge features."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_edge_features: bool = False
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_features = use_edge_features
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.o_linear = nn.Linear(hidden_dim, hidden_dim)
        
        if use_edge_features:
            self.edge_proj = nn.Linear(hidden_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        
        # Ensure x has the right shape
        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        assert x.size(1) == self.hidden_dim, f"Hidden dim mismatch: {x.size(1)} vs {self.hidden_dim}"
        
        # Linear projections in batch from hidden_dim => h * head_dim
        Q = self.q_linear(x).reshape(num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).reshape(num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).reshape(num_nodes, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('nhd,mhd->nmh', Q, K) / self.scale
        
        # Apply edge features as bias if available
        if self.use_edge_features and edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)  # [num_edges, num_heads]
            # Add edge bias to corresponding positions in attention scores
            for idx, (i, j) in enumerate(edge_index.T):
                scores[i, j, :] += edge_bias[idx]
        
        # Mask attention scores
        scores = scores.transpose(-1, -2)  # [num_nodes, num_heads, num_nodes]
        mask = ~adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
        scores.masked_fill_(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.einsum('nhm,mhd->nhd', attn_weights, V)
        context = context.reshape(num_nodes, self.hidden_dim)
        
        # Final linear layer
        output = self.o_linear(context)
        
        return output


class GlobalGraphPool(nn.Module):
    """Global pooling to aggregate node features into graph representation."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment for each node (for batched graphs)
        
        Returns:
            Graph representation [batch_size, hidden_dim * 3]
        """
        if batch is None:
            # Single graph case
            # Mean pooling
            mean_pool = x.mean(dim=0, keepdim=True)
            
            # Max pooling
            max_pool = x.max(dim=0, keepdim=True)[0]
            
            # Attention-weighted pooling
            attn_weights = F.softmax(self.attention_pool(x), dim=0)
            attn_pool = (x * attn_weights).sum(dim=0, keepdim=True)
            
            # Concatenate different pooling strategies
            graph_repr = torch.cat([mean_pool, max_pool, attn_pool], dim=1)
        else:
            # Batched graphs case
            batch_size = batch.max().item() + 1
            mean_pool = torch.zeros(batch_size, x.size(1), device=x.device)
            max_pool = torch.zeros(batch_size, x.size(1), device=x.device)
            attn_pool = torch.zeros(batch_size, x.size(1), device=x.device)
            
            for b in range(batch_size):
                mask = (batch == b)
                batch_x = x[mask]
                
                mean_pool[b] = batch_x.mean(dim=0)
                max_pool[b] = batch_x.max(dim=0)[0]
                
                attn_weights = F.softmax(self.attention_pool(batch_x), dim=0)
                attn_pool[b] = (batch_x * attn_weights).sum(dim=0)
            
            graph_repr = torch.cat([mean_pool, max_pool, attn_pool], dim=1)
        
        return graph_repr

class GraphTransformer(nn.Module):
    """
    Graph Transformer model that processes graphs with variable numbers of nodes
    and edges to produce a fixed-size 1280-dimensional representation.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1280,
        max_nodes: int = 1000,
        use_edge_features: bool = False,
        edge_feat_dim: Optional[int] = None
    ):
        """
        Args:
            node_feat_dim: Dimension of input node features
            hidden_dim: Hidden dimension for transformer layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            output_dim: Final output dimension (default 1280)
            max_nodes: Maximum number of nodes for positional encoding
            use_edge_features: Whether to use edge features
            edge_feat_dim: Dimension of edge features (if used)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features
        
        # Node feature projection
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        
        # Edge feature projection (optional)
        if use_edge_features and edge_feat_dim:
            self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_nodes, hidden_dim))
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_features=use_edge_features
            )
            for _ in range(num_layers)
        ])
        
        # Global graph pooling layers
        self.global_pool = GlobalGraphPool(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # *3 for mean, max, and attention pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Graph Transformer.
        
        Args:
            node_features: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_feat_dim] (optional)
            batch: Batch assignment for each node [num_nodes] (for batched graphs)
        
        Returns:
            Graph representation [batch_size, 1280]
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Add positional embeddings
        if num_nodes <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :num_nodes, :]  # [1, num_nodes, hidden_dim]
            x = x + pos_emb.squeeze(0)  # Broadcasting: [num_nodes, hidden_dim]
        else:
            # If graph is larger than max_nodes, use sinusoidal encoding
            pos_enc = self.get_sinusoidal_encoding(num_nodes, self.hidden_dim).to(device)
            x = x + pos_enc
        
        x = self.dropout(x)
        
        # Encode edge features if provided
        edge_attr = None
        if self.use_edge_features and edge_features is not None:
            edge_attr = self.edge_encoder(edge_features)
        
        # Create adjacency mask from edge_index
        adj_mask = self.create_adjacency_mask(edge_index, num_nodes)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, adj_mask, edge_index, edge_attr)
        
        x = self.layer_norm(x)
        
        # Global pooling to get graph-level representation
        graph_repr = self.global_pool(x, batch)
        
        # Final projection to output dimension
        output = self.output_projection(graph_repr)
        
        return output
    
    def create_adjacency_mask(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency mask for attention mechanism."""
        device = edge_index.device
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        
        if edge_index.size(1) > 0:  # Check if there are edges
            # Ensure indices are within bounds
            src = edge_index[0].clamp(0, num_nodes - 1)
            dst = edge_index[1].clamp(0, num_nodes - 1)
            
            mask[src, dst] = True
            mask[dst, src] = True  # Assuming undirected graph
        
        # Add self-loops
        mask.fill_diagonal_(True)
        return mask
    
    def get_sinusoidal_encoding(self, num_positions: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pos_encoding = torch.zeros(num_positions, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding

def main():
    # Initialize model
    model = GraphTransformer(
        node_feat_dim=64,  # Input node feature dimension
        hidden_dim=256,     # Hidden dimension
        num_heads=8,        # Number of attention heads
        num_layers=4,       # Number of transformer layers
        dropout=0.1,
        output_dim=1280,    # Output dimension
        use_edge_features=False  # Set to True if you have edge features
    )
    
    # Create sample graph data
    num_nodes = 20
    node_features = torch.randn(num_nodes, 64)  # Random node features
    
    # Create valid edge indices (simple ring graph + some random edges)
    edges = []
    # Ring connectivity
    for i in range(num_nodes):
        edges.append([i, (i + 1) % num_nodes])
    # Add some random edges
    for _ in range(10):
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        if src != dst:
            edges.append([src, dst])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Forward pass
    output = model(node_features, edge_index)
    print(f"Output shape: {output.shape}")  # Should be [1, 1280]
    
    # Example with batched graphs
    batch = torch.tensor([0]*10 + [1]*10, dtype=torch.long)  # Two graphs with 10 nodes each
    output_batched = model(node_features, edge_index, batch=batch)
    print(f"Batched output shape: {output_batched.shape}")  # Should be [2, 1280]

    # Augmentations to apply in pipeline
    # each augmentation is 80% for student, 20% for teacher
    # 1. sample a subset of the graph via random walks
    # 2. remove a subset of edges and keep all nodes
    # 3. teacher keeps all 

if __name__ == "__main__":
    main()