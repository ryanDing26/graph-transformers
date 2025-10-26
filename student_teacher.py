import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from graph_transformer import GraphTransformer


class StudentTeacherGraphTransformer(nn.Module):
    """
    Wrapper class that maintains student and teacher GraphTransformer models
    with Exponential Moving Average (EMA) updates for the teacher.
    """
    
    def __init__(
        self,
        # GraphTransformer parameters
        node_feat_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1280,
        max_nodes: int = 1000,
        use_edge_features: bool = False,
        edge_feat_dim: Optional[int] = None,
        # EMA parameters
        ema_momentum: float = 0.996,
        use_teacher_for_inference: bool = True
    ):
        """
        Parameters
        ----------
        node_feat_dim: int
            Dimension of input node features
        hidden_dim: int 
            Hidden dimension for transformer layers
        num_heads: int 
            Number of attention heads
        num_layers: int 
            Number of transformer layers
        dropout: float 
            Dropout probability
        output_dim: int 
            Final output dimension (default 1280)
        max_nodes: int 
            Maximum number of nodes for positional encoding
        use_edge_features: bool
            Whether to use edge features
        edge_feat_dim: int | None
            Dimension of edge features (if used)
        ema_momentum: float
            Momentum coefficient for EMA updates (default 0.996)
        use_teacher_for_inference: bool
            Whether to use teacher model during evaluation
        """
        super().__init__()
        
        self.ema_momentum = ema_momentum
        self.use_teacher_for_inference = use_teacher_for_inference
        
        # Initialize student model
        self.student = GraphTransformer(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_dim,
            max_nodes=max_nodes,
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim
        )
        
        # Initialize teacher model as a copy of student
        self.teacher = copy.deepcopy(self.student)
        
        # Freeze teacher parameters (they're updated via EMA, not gradients)
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Initialize iteration counter for momentum scheduling (optional)
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        
    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        """Update teacher model parameters using EMA of student parameters.
        
        Parameters
        ----------
        momentum: float | None 
            Optional momentum value (uses self.ema_momentum if None)
        """
        if momentum is None:
            momentum = self.ema_momentum
            
        # Update teacher parameters based on EMA of student parameters
        for student_param, teacher_param in zip(
            self.student.parameters(), 
            self.teacher.parameters()
        ):
            teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
            
        # Update batch norm statistics if any
        for student_module, teacher_module in zip(
            self.student.modules(), 
            self.teacher.modules()
        ):
            if isinstance(student_module, nn.BatchNorm1d):
                teacher_module.running_mean = momentum * teacher_module.running_mean + \
                                             (1 - momentum) * student_module.running_mean
                teacher_module.running_var = momentum * teacher_module.running_var + \
                                            (1 - momentum) * student_module.running_var
                teacher_module.num_batches_tracked = student_module.num_batches_tracked
                
        self.num_updates += 1
    
    @torch.no_grad()
    def momentum_schedule(self, base_momentum: float = 0.996, max_momentum: float = 0.999, 
                          epochs: int = 100, current_epoch: int = 0) -> float:
        """
        Cosine schedule for momentum coefficient (optional).
        Gradually increases momentum during training.
        
        Parameters
        ----------
        base_momentum: float
            Starting momentum
        max_momentum: float
            Maximum momentum
        epochs: int
            Total number of epochs
        current_epoch: int
            Current epoch
            
        Returns
        -------
        float
            Scheduled momentum value
        """
        return max_momentum - (max_momentum - base_momentum) * \
               (1 + torch.cos(torch.tensor(torch.pi * current_epoch / epochs))) / 2
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_both: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through student and/or teacher model.
        
        Args:
            ... (same as GraphTransformer.forward)
            return_both: If True, returns both student and teacher outputs
            
        Returns:
            If training or return_both=True: student output or (student, teacher) tuple
            If eval and use_teacher_for_inference: teacher output
        """
        if return_both:
            student_out = self.student(node_features, edge_index, edge_features, batch)
            with torch.no_grad():
                teacher_out = self.teacher(node_features, edge_index, edge_features, batch)
            return student_out, teacher_out
        
        if self.training:
            # During training, use student model
            return self.student(node_features, edge_index, edge_features, batch)
        else:
            # During evaluation, use teacher model if specified
            if self.use_teacher_for_inference:
                return self.teacher(node_features, edge_index, edge_features, batch)
            else:
                return self.student(node_features, edge_index, edge_features, batch)
    
    def get_student_params(self) -> torch.nn.parameter.Parameter:
        """Get student model parameters for optimizer."""
        return self.student.parameters()
    
    def get_representations(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get representations from both student and teacher models.
        Useful for contrastive learning or distillation losses.
        """
        student_repr = self.student(node_features, edge_index, edge_features, batch)
        with torch.no_grad():
            teacher_repr = self.teacher(node_features, edge_index, edge_features, batch)
        
        return {
            'student': student_repr,
            'teacher': teacher_repr
        }

# Training utilities
class EMAUpdater:
    """
    Utility class to handle EMA updates with different strategies.
    """
    
    def __init__(
        self,
        model: StudentTeacherGraphTransformer,
        momentum: float = 0.996,
        update_after_step: int = 0,
        update_every_n_steps: int = 1
    ):
        """
        Args:
            model: StudentTeacherGraphTransformer instance
            momentum: Base momentum for EMA
            update_after_step: Start updating after this many steps
            update_every_n_steps: Update teacher every n steps
        """
        self.model = model
        self.momentum = momentum
        self.update_after_step = update_after_step
        self.update_every_n_steps = update_every_n_steps
        self.step = 0
        
    def update(self, momentum: Optional[float] = None):
        """Update teacher model if conditions are met."""
        self.step += 1
        
        if self.step > self.update_after_step and \
           self.step % self.update_every_n_steps == 0:
            self.model.update_teacher(momentum or self.momentum)
    
    def set_momentum(self, momentum: float):
        """Update the momentum value."""
        self.momentum = momentum

# Example training loop
def train_with_ema(
    model: StudentTeacherGraphTransformer,
    data_loader,
    optimizer,
    num_epochs: int = 100,
    device: str = 'cuda'
):
    """
    Example training loop with EMA updates.
    """
    model = model.to(device)
    ema_updater = EMAUpdater(model, momentum=0.996)
    
    for epoch in range(num_epochs):
        # Optional: Use momentum schedule
        current_momentum = model.momentum_schedule(
            current_epoch=epoch,
            epochs=num_epochs
        )
        ema_updater.set_momentum(current_momentum)
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch.get('edge_features')
            if edge_features is not None:
                edge_features = edge_features.to(device)
            
            # Get representations from both models
            representations = model.get_representations(
                node_features, edge_index, edge_features
            )
            
            student_repr = representations['student']
            teacher_repr = representations['teacher']
            
            # Example: Contrastive loss between student and teacher
            # You can implement your specific loss here
            loss = F.mse_loss(student_repr, teacher_repr.detach())
            
            # Standard training steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher model with EMA
            ema_updater.update()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], "
                      f"Step [{batch_idx}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Momentum: {current_momentum:.4f}")


# Alternative: Manual update in training loop
def simple_training_example():
    """
    Simple example showing manual EMA updates.
    """
    # Initialize model
    model = StudentTeacherGraphTransformer(
        node_feat_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        output_dim=1280,
        ema_momentum=0.996
    )
    
    # Only optimize student parameters
    optimizer = torch.optim.Adam(model.get_student_params(), lr=1e-4)
    
    # Training loop
    for epoch in range(100):
        for batch in data_loader:
            # Forward pass through student
            student_output = model.student(
                batch['node_features'],
                batch['edge_index']
            )
            
            # Compute loss (example)
            loss = compute_loss(student_output, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher with EMA
            model.update_teacher()
            
    # During inference, the model will automatically use teacher
    model.eval()
    with torch.no_grad():
        inference_output = model(test_features, test_edges)


# Utility function to check parameter differences
@torch.no_grad()
def check_parameter_differences(model: StudentTeacherGraphTransformer):
    """
    Check the parameter differences between student and teacher models.
    Useful for debugging and monitoring convergence.
    """
    total_diff = 0.0
    param_count = 0
    
    for student_param, teacher_param in zip(
        model.student.parameters(),
        model.teacher.parameters()
    ):
        diff = (student_param - teacher_param).abs().mean().item()
        total_diff += diff
        param_count += 1
    
    avg_diff = total_diff / param_count if param_count > 0 else 0.0
    return {
        'average_param_diff': avg_diff,
        'total_diff': total_diff,
        'num_parameters': param_count
    }


def main():
    # Create model with EMA
    model = StudentTeacherGraphTransformer(
        node_feat_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        output_dim=1280,
        ema_momentum=0.996,
        use_teacher_for_inference=True
    )
    
    # Create dummy data
    batch_size = 32
    num_nodes = 100
    num_edges = 300
    
    node_features = torch.randn(num_nodes, 128)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Get outputs from both models
    student_out, teacher_out = model(
        node_features, 
        edge_index, 
        return_both=True
    )
    
    print(f"Student output shape: {student_out.shape}")
    print(f"Teacher output shape: {teacher_out.shape}")
    
    # Check initial parameter differences (should be 0)
    diff_stats = check_parameter_differences(model)
    print(f"Initial parameter difference: {diff_stats['average_param_diff']:.6f}")
    
    # Simulate some updates
    for _ in range(10):
        # Random perturbation to student parameters
        for param in model.student.parameters():
            param.data += torch.randn_like(param) * 0.01
        
        # Update teacher
        model.update_teacher()
    
    # Check parameter differences after updates
    diff_stats = check_parameter_differences(model)
    print(f"Parameter difference after updates: {diff_stats['average_param_diff']:.6f}")

if __name__ == "__main__":
    main()