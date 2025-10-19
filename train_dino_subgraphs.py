"""
DINO Training Pipeline for Large Nucleus Graphs
Trains on sampled subgraphs with student-teacher architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import pickle

# Import your existing modules
import sys
sys.path.append('models')
from student_teacher import StudentTeacherGraphTransformer
from dino_algorithms import (
    DINOHead, DINOLoss, GraphAugmentation, 
    dino_training_step
)
from subgraph_sampler import SubgraphSampler, GraphDataLoader


class DINOTrainer:
    """
    DINO trainer for large nucleus graphs using subgraph sampling.
    """
    
    def __init__(
        self,
        model: StudentTeacherGraphTransformer,
        student_head: DINOHead,
        teacher_head: DINOHead,
        dino_loss: DINOLoss,
        augmentor: GraphAugmentation,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.04,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        ema_momentum_schedule: bool = True,
        log_dir: str = 'logs/dino'
    ):
        """
        Args:
            model: StudentTeacherGraphTransformer
            student_head: DINO projection head for student
            teacher_head: DINO projection head for teacher
            dino_loss: DINO loss module
            augmentor: Graph augmentation module
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            warmup_epochs: Learning rate warmup epochs
            max_epochs: Total training epochs
            ema_momentum_schedule: Use cosine schedule for EMA momentum
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.student_head = student_head.to(device)
        self.teacher_head = teacher_head.to(device)
        self.dino_loss = dino_loss.to(device)
        self.augmentor = augmentor
        self.device = device
        
        # Optimizer for student only
        params = list(model.student.parameters()) + list(student_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = learning_rate
        
        # EMA momentum schedule
        self.ema_momentum_schedule = ema_momentum_schedule
        self.base_momentum = model.ema_momentum
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
    def train_epoch(
        self,
        dataloader: GraphDataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: GraphDataLoader providing batched subgraphs
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Update learning rate
        lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Update EMA momentum
        momentum = self._get_momentum(epoch)
        self.model.ema_momentum = momentum
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            node_features = torch.from_numpy(batch['node_features']).float().to(self.device)
            edge_index = torch.from_numpy(batch['edge_index']).long().to(self.device).t()
            batch_indices = torch.from_numpy(batch['batch']).long().to(self.device)
            
            edge_attr = None
            if 'edge_attr' in batch and batch['edge_attr'] is not None:
                edge_attr = torch.from_numpy(batch['edge_attr']).float().to(self.device)
            
            # DINO training step with augmentations
            loss = self._dino_step(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_indices,
                epoch=epoch
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.student.parameters()) + 
                list(self.student_head.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update teacher with EMA
            self.model.update_teacher(momentum)
            
            # Update teacher head with EMA
            with torch.no_grad():
                for s_param, t_param in zip(
                    self.student_head.parameters(),
                    self.teacher_head.parameters()
                ):
                    t_param.data = momentum * t_param.data + \
                                  (1 - momentum) * s_param.data
            
            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': lr,
                'momentum': momentum
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
                self.writer.add_scalar('train/momentum', momentum, self.global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'lr': lr,
            'momentum': momentum
        }
    
    def _dino_step(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """
        Single DINO training step with graph augmentations.
        """
        # Convert to numpy for augmentation (your augmentation is numpy-based)
        node_feat_np = node_features.cpu().numpy()
        edge_idx_np = edge_index.t().cpu().numpy()
        
        # Create two augmented views
        view1 = self.augmentor.create_views(
            torch.from_numpy(node_feat_np).to(self.device),
            torch.from_numpy(edge_idx_np).to(self.device),
            num_views=1
        )[0]
        
        view2 = self.augmentor.create_views(
            torch.from_numpy(node_feat_np).to(self.device),
            torch.from_numpy(edge_idx_np).to(self.device),
            num_views=1
        )[0]
        
        # Student forward on view1
        student_embed = self.model.student(
            view1[0], view1[1], edge_attr, batch
        )
        student_proj = self.student_head(student_embed)
        
        # Teacher forward on view2 (no gradients)
        with torch.no_grad():
            teacher_embed = self.model.teacher(
                view2[0], view2[1], edge_attr, batch
            )
            teacher_proj = self.teacher_head(teacher_embed)
        
        # DINO loss
        loss = self.dino_loss(student_proj, teacher_proj, epoch)
        
        # Symmetric loss (also predict view2 from view1)
        student_embed_2 = self.model.student(
            view2[0], view2[1], edge_attr, batch
        )
        student_proj_2 = self.student_head(student_embed_2)
        
        with torch.no_grad():
            teacher_embed_1 = self.model.teacher(
                view1[0], view1[1], edge_attr, batch
            )
            teacher_proj_1 = self.teacher_head(teacher_embed_1)
        
        loss_2 = self.dino_loss(student_proj_2, teacher_proj_1, epoch)
        
        # Total loss
        total_loss = (loss + loss_2) / 2
        
        return total_loss
    
    def _get_lr(self, epoch: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if epoch < self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def _get_momentum(self, epoch: int) -> float:
        """Cosine momentum schedule (increases from base to 1.0)."""
        if not self.ema_momentum_schedule:
            return self.base_momentum
        
        max_momentum = 0.999
        progress = epoch / self.max_epochs
        return max_momentum - (max_momentum - self.base_momentum) * \
               0.5 * (1 + np.cos(np.pi * progress))
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'student_head_state_dict': self.student_head.state_dict(),
            'teacher_head_state_dict': self.teacher_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dino_loss_state_dict': self.dino_loss.state_dict(),
            'global_step': self.global_step,
            'metrics': metrics
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.student_head.load_state_dict(checkpoint['student_head_state_dict'])
        self.teacher_head.load_state_dict(checkpoint['teacher_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dino_loss.load_state_dict(checkpoint['dino_loss_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"✓ Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint['epoch']


class InferenceEngine:
    """
    Generate embeddings for large graphs using trained DINO model.
    Uses sliding window approach with aggregation.
    """
    
    def __init__(
        self,
        model: StudentTeacherGraphTransformer,
        sampler: SubgraphSampler,
        device: str = 'cuda',
        use_teacher: bool = True
    ):
        """
        Args:
            model: Trained StudentTeacherGraphTransformer
            sampler: SubgraphSampler for consistent sampling
            device: Device for inference
            use_teacher: Use teacher model (recommended)
        """
        self.model = model.to(device)
        self.model.eval()
        self.sampler = sampler
        self.device = device
        self.use_teacher = use_teacher
    
    @torch.no_grad()
    def embed_graph(
        self,
        graph_data: Dict,
        num_samples: int = 10,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Generate embedding for a large graph by sampling and aggregating subgraphs.
        
        Args:
            graph_data: Full graph dictionary
            num_samples: Number of subgraphs to sample
            aggregation: How to aggregate ('mean', 'max', 'concat')
            
        Returns:
            Graph embedding (1280-dim or larger if concat)
        """
        # Sample subgraphs
        subgraphs = self.sampler.sample_subgraphs(
            graph_data, 
            num_samples=num_samples
        )
        
        embeddings = []
        
        for subgraph in subgraphs:
            # Convert to tensors
            node_features = torch.from_numpy(subgraph['node_features']).float().to(self.device)
            edge_index = torch.from_numpy(subgraph['edge_index']).long().to(self.device).t()
            
            edge_attr = None
            if subgraph['edge_attr'] is not None:
                edge_attr = torch.from_numpy(subgraph['edge_attr']).float().to(self.device)
            
            # Get embedding
            if self.use_teacher:
                embed = self.model.teacher(node_features, edge_index, edge_attr)
            else:
                embed = self.model.student(node_features, edge_index, edge_attr)
            
            embeddings.append(embed.cpu().numpy())
        
        embeddings = np.array(embeddings)  # (num_samples, 1280)
        
        # Aggregate
        if aggregation == 'mean':
            return embeddings.mean(axis=0)
        elif aggregation == 'max':
            return embeddings.max(axis=0)
        elif aggregation == 'concat':
            return embeddings.flatten()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def main():
    parser = argparse.ArgumentParser(description='Train DINO on nucleus graphs')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing graph .pkl files')
    parser.add_argument('--output_dir', type=str, default='outputs/dino',
                       help='Output directory for checkpoints')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=1280)
    
    # DINO parameters
    parser.add_argument('--dino_out_dim', type=int, default=65536)
    parser.add_argument('--teacher_temp', type=float, default=0.04)
    parser.add_argument('--student_temp', type=float, default=0.1)
    
    # Subgraph sampling
    parser.add_argument('--sampling_strategy', type=str, default='spatial',
                       choices=['spatial', 'khop', 'random_walk'])
    parser.add_argument('--subgraph_size', type=int, default=1000)
    parser.add_argument('--subgraphs_per_graph', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find graph files
    data_dir = Path(args.data_dir)
    graph_files = list(data_dir.glob('*/nucleus_graph.pkl'))
    print(f"Found {len(graph_files)} graph files")
    
    if len(graph_files) == 0:
        print("Error: No graph files found!")
        return
    
    # Load first graph to get dimensions
    with open(graph_files[0], 'rb') as f:
        sample_graph = pickle.load(f)
    
    node_feat_dim = sample_graph['node_features'].shape[1]
    print(f"Node feature dimension: {node_feat_dim}")
    
    # Create model
    print("\nInitializing models...")
    model = StudentTeacherGraphTransformer(
        node_feat_dim=node_feat_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        use_edge_features=False,
        ema_momentum=0.996
    )
    
    # Create DINO heads
    student_head = DINOHead(
        in_dim=args.output_dim,
        out_dim=args.dino_out_dim
    )
    
    teacher_head = DINOHead(
        in_dim=args.output_dim,
        out_dim=args.dino_out_dim
    )
    
    # Copy student head to teacher
    teacher_head.load_state_dict(student_head.state_dict())
    for param in teacher_head.parameters():
        param.requires_grad = False
    
    # Create DINO loss
    dino_loss = DINOLoss(
        out_dim=args.dino_out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp
    )
    
    # Create augmentor
    augmentor = GraphAugmentation(
        drop_node_prob=0.1,
        drop_edge_prob=0.2,
        mask_feat_prob=0.15,
        subgraph_prob=0.3
    )
    
    # Create subgraph sampler
    print("\nSetting up subgraph sampler...")
    sampler = SubgraphSampler(
        strategy=args.sampling_strategy,
        subgraph_size=args.subgraph_size,
        overlap=0.2,
        min_nodes=200,
        max_nodes=2000
    )
    
    # Create dataloader
    dataloader = GraphDataLoader(
        graph_paths=[str(p) for p in graph_files],
        sampler=sampler,
        subgraphs_per_graph=args.subgraphs_per_graph,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    print(f"Dataloader: {len(dataloader)} batches per epoch")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = DINOTrainer(
        model=model,
        student_head=student_head,
        teacher_head=teacher_head,
        dino_loss=dino_loss,
        augmentor=augmentor,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        log_dir=str(Path(args.output_dir) / 'logs')
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  LR: {metrics['lr']:.6f}")
        print(f"  Momentum: {metrics['momentum']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch, metrics)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()