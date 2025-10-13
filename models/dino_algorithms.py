import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from student_teacher import StudentTeacherGraphTransformer

class DINOHead(nn.Module):
    """
    Projection head for DINO training. Projects graph embeddings to a lower
    dimensional space where the DINO loss is computed.
    """
    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,  # High-dim output for DINO
        norm_last_layer: bool = True,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Last layer with optional L2 normalization
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """
    DINO loss with centering and sharpening.
    """
    def __init__(
        self,
        out_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 10,
        student_temp: float = 0.1,
        center_momentum: float = 0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.center_momentum = center_momentum
        
        # Centering vector to avoid mode collapse
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int = 0
    ) -> torch.Tensor:
        """
        Compute DINO loss between student and teacher outputs.
        
        Args:
            student_output: Student projections [batch_size, out_dim]
            teacher_output: Teacher projections [batch_size, out_dim]
            epoch: Current epoch for temperature scheduling
        """
        # Temperature scheduling for teacher
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.warmup_teacher_temp + \
                (self.teacher_temp - self.warmup_teacher_temp) * \
                (epoch / self.warmup_teacher_temp_epochs)
        else:
            teacher_temp = self.teacher_temp
            
        # Softmax with temperature
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        
        # Center and sharpen teacher predictions
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()  # Stop gradient
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        # Update center with EMA
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center vector with exponential moving average."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)


class GraphAugmentation:
    """
    Graph-specific augmentations for DINO training.
    Creates multiple views of the same graph.
    """
    def __init__(
        self,
        drop_node_prob: float = 0.1,
        drop_edge_prob: float = 0.2,
        mask_feat_prob: float = 0.15,
        subgraph_prob: float = 0.3
    ):
        self.drop_node_prob = drop_node_prob
        self.drop_edge_prob = drop_edge_prob
        self.mask_feat_prob = mask_feat_prob
        self.subgraph_prob = subgraph_prob
    
    def drop_nodes(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        drop_prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop nodes from graph."""
        num_nodes = node_features.size(0)
        keep_mask = torch.rand(num_nodes) > drop_prob
        keep_indices = torch.where(keep_mask)[0]
        
        # Remap node features
        new_features = node_features[keep_mask]
        
        # Remap edges
        node_map = torch.full((num_nodes,), -1, dtype=torch.long)
        node_map[keep_indices] = torch.arange(len(keep_indices))
        
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        new_edge_index = node_map[edge_index[:, edge_mask]]
        
        return new_features, new_edge_index
    
    def drop_edges(
        self,
        edge_index: torch.Tensor,
        drop_prob: float
    ) -> torch.Tensor:
        """Randomly drop edges from graph."""
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges) > drop_prob
        return edge_index[:, keep_mask]
    
    def mask_features(
        self,
        node_features: torch.Tensor,
        mask_prob: float
    ) -> torch.Tensor:
        """Randomly mask node features."""
        mask = torch.rand_like(node_features) > mask_prob
        # Replace masked features with zeros or random values
        masked_features = node_features * mask
        return masked_features
    
    def create_views(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_views: int = 2
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create multiple augmented views of the graph."""
        views = []
        
        for _ in range(num_views):
            # Apply augmentations with some probability
            aug_features = node_features.clone()
            aug_edges = edge_index.clone()
            
            # Node dropping
            if torch.rand(1) < self.subgraph_prob:
                aug_features, aug_edges = self.drop_nodes(
                    aug_features, aug_edges, self.drop_node_prob
                )
            
            # Edge dropping  
            if torch.rand(1) < 0.5:
                aug_edges = self.drop_edges(aug_edges, self.drop_edge_prob)
            
            # Feature masking
            if torch.rand(1) < 0.5:
                aug_features = self.mask_features(aug_features, self.mask_feat_prob)
            
            views.append((aug_features, aug_edges))
        
        return views


# Integration function to add DINO to your existing model
def add_dino_to_student_teacher(
    student_teacher_model: 'StudentTeacherGraphTransformer',
    out_dim: int = 65536
) -> Tuple[StudentTeacherGraphTransformer, DINOHead, DINOHead, DINOLoss]:
    """
    Add DINO heads and loss to existing Student-Teacher model.
    
    Returns:
        - Modified model
        - Student DINO head
        - Teacher DINO head  
        - DINO loss module
    """
    # Get output dimension from model
    output_dim = student_teacher_model.student.output_projection[-1].out_features
    
    # Create DINO heads
    student_head = DINOHead(in_dim=output_dim, out_dim=out_dim)
    teacher_head = DINOHead(in_dim=output_dim, out_dim=out_dim)
    
    # Initialize teacher head with student weights
    teacher_head.load_state_dict(student_head.state_dict())
    
    # Freeze teacher head
    for param in teacher_head.parameters():
        param.requires_grad = False
    
    # Create DINO loss
    dino_loss = DINOLoss(out_dim=out_dim)
    
    return student_teacher_model, student_head, teacher_head, dino_loss


# Training step example
def dino_training_step(
    model: StudentTeacherGraphTransformer,
    student_head: DINOHead,
    teacher_head: DINOHead,
    dino_loss: DINOLoss,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    augmentor: GraphAugmentation,
    epoch: int = 0
) -> torch.Tensor:
    """
    Single DINO training step.
    """
    # Create augmented views
    views = augmentor.create_views(node_features, edge_index, num_views=2)
    
    total_loss = 0
    
    # Process each view pair
    for i, (feat1, edge1) in enumerate(views):
        for j, (feat2, edge2) in enumerate(views):
            if i == j:
                continue
                
            # Get embeddings from backbone
            student_embed = model.student(feat1, edge1)
            with torch.no_grad():
                teacher_embed = model.teacher(feat2, edge2)
            
            # Project through DINO heads
            student_proj = student_head(student_embed)
            teacher_proj = teacher_head(teacher_embed)
            
            # Compute DINO loss
            loss = dino_loss(student_proj, teacher_proj, epoch)
            total_loss += loss
    
    # Average over all view pairs
    total_loss = total_loss / (len(views) * (len(views) - 1))
    
    # Update teacher head with EMA (same momentum as backbone)
    with torch.no_grad():
        for student_param, teacher_param in zip(
            student_head.parameters(),
            teacher_head.parameters()
        ):
            teacher_param.data = model.ema_momentum * teacher_param.data + \
                               (1 - model.ema_momentum) * student_param.data
    
    return total_loss