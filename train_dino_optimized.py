"""
MEMORY-OPTIMIZED DINO Training for Large Nucleus Graphs
Uses mixed precision, gradient checkpointing, and smaller batch sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import pickle
import gc

# Import existing modules
import sys
sys.path.append('models')
from student_teacher import StudentTeacherGraphTransformer
from dino_algorithms import DINOHead, DINOLoss
from subgraph_sampler import SubgraphSampler, GraphDataLoader


class MemoryEfficientDINOTrainer:
    """
    Memory-optimized DINO trainer with:
    - Mixed precision (FP16) training
    - Gradient accumulation
    - Periodic garbage collection
    """
    
    def __init__(
        self,
        model: StudentTeacherGraphTransformer,
        student_head: DINOHead,
        teacher_head: DINOHead,
        dino_loss: DINOLoss,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.04,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        ema_momentum_schedule: bool = True,
        log_dir: str = 'logs/dino',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.device = device
        
        # Initialize models on CPU first, then move to GPU
        print("Initializing models on CPU...")
        self.model = model
        self.student_head = student_head
        self.teacher_head = teacher_head
        
        # Move to GPU one by one to monitor memory
        print(f"Moving models to {device}...")
        try:
            self.model = self.model.to(device)
            print(f"  ✓ Main model on GPU")
            
            self.student_head = self.student_head.to(device)
            print(f"  ✓ Student head on GPU")
            
            self.teacher_head = self.teacher_head.to(device)
            print(f"  ✓ Teacher head on GPU")
            
        except RuntimeError as e:
            print(f"\n✗ GPU OOM Error: {e}")
            print("\nTrying to free memory...")
            torch.cuda.empty_cache()
            gc.collect()
            raise
        
        self.dino_loss = dino_loss.to(device)
        
        # Mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer
        params = list(model.student.parameters()) + list(student_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduling
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = learning_rate
        self.ema_momentum_schedule = ema_momentum_schedule
        self.base_momentum = model.ema_momentum
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
        print(f"✓ Trainer initialized")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
        
    def train_epoch(
        self,
        dataloader: GraphDataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with memory optimizations."""
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
        
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device
                node_features = torch.from_numpy(batch['node_features']).float().to(self.device)
                edge_index = torch.from_numpy(batch['edge_index']).long().to(self.device).t()
                batch_indices = torch.from_numpy(batch['batch']).long().to(self.device)
                
                edge_attr = None
                if 'edge_attr' in batch and batch['edge_attr'] is not None:
                    edge_attr = torch.from_numpy(batch['edge_attr']).float().to(self.device)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    loss = self._dino_step(
                        node_features=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        batch=batch_indices,
                        epoch=epoch
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_counter += 1
                
                # Update weights after accumulation steps
                if accumulation_counter >= self.gradient_accumulation_steps:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student.parameters()) + 
                            list(self.student_head.parameters()),
                            max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.student.parameters()) + 
                            list(self.student_head.parameters()),
                            max_norm=1.0
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Update teacher with EMA
                    self.model.update_teacher(momentum)
                    
                    # Update teacher head
                    with torch.no_grad():
                        for s_param, t_param in zip(
                            self.student_head.parameters(),
                            self.teacher_head.parameters()
                        ):
                            t_param.data = momentum * t_param.data + \
                                          (1 - momentum) * s_param.data
                    
                    accumulation_counter = 0
                
                # Logging
                actual_loss = loss.item() * self.gradient_accumulation_steps
                epoch_loss += actual_loss
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': actual_loss,
                    'lr': lr,
                    'momentum': momentum
                })
                
                # Periodic garbage collection
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Log to tensorboard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('train/loss', actual_loss, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
                    self.writer.add_scalar('train/momentum', momentum, self.global_step)
                    
                    # Log memory usage
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        self.writer.add_scalar('system/gpu_memory_gb', mem_allocated, self.global_step)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠ OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise
        
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
        """Single DINO training step with batch-safe augmentations."""
        # Batch-safe augmentations (no node dropping)
        view1_feat = self._augment_features(node_features)
        view1_edges = self._augment_edges(edge_index)
        
        view2_feat = self._augment_features(node_features)
        view2_edges = self._augment_edges(edge_index)
        
        # Student forward on view1
        student_embed = self.model.student(
            view1_feat, view1_edges, edge_attr, batch
        )
        student_proj = self.student_head(student_embed)
        
        # Teacher forward on view2 (no gradients)
        with torch.no_grad():
            teacher_embed = self.model.teacher(
                view2_feat, view2_edges, edge_attr, batch
            )
            teacher_proj = self.teacher_head(teacher_embed)
        
        # DINO loss
        loss = self.dino_loss(student_proj, teacher_proj, epoch)
        
        # Symmetric loss
        student_embed_2 = self.model.student(
            view2_feat, view2_edges, edge_attr, batch
        )
        student_proj_2 = self.student_head(student_embed_2)
        
        with torch.no_grad():
            teacher_embed_1 = self.model.teacher(
                view1_feat, view1_edges, edge_attr, batch
            )
            teacher_proj_1 = self.teacher_head(teacher_embed_1)
        
        loss_2 = self.dino_loss(student_proj_2, teacher_proj_1, epoch)
        
        return (loss + loss_2) / 2
    
    def _augment_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """Augment node features without changing graph structure."""
        aug_features = node_features.clone()
        
        # 1. Feature masking (80% chance)
        if torch.rand(1).item() < 0.8:
            mask = torch.rand_like(aug_features) > 0.15
            aug_features = aug_features * mask
        
        # 2. Add Gaussian noise (50% chance)
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(aug_features) * 0.1
            aug_features = aug_features + noise
        
        return aug_features
    
    def _augment_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Augment edges without changing node count."""
        # Drop edges (80% chance)
        if torch.rand(1).item() < 0.8:
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=edge_index.device) > 0.2
            return edge_index[:, keep_mask]
        return edge_index
    
    def _get_lr(self, epoch: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if epoch < self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def _get_momentum(self, epoch: int) -> float:
        """Cosine momentum schedule."""
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
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='Memory-optimized DINO training')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/dino')
    
    # Model architecture (REDUCED DEFAULTS)
    parser.add_argument('--hidden_dim', type=int, default=128)  # Reduced from 256
    parser.add_argument('--num_heads', type=int, default=4)    # Reduced from 8
    parser.add_argument('--num_layers', type=int, default=3)   # Reduced from 4
    parser.add_argument('--output_dim', type=int, default=1280)
    
    # DINO parameters (REDUCED DEFAULTS)
    parser.add_argument('--dino_out_dim', type=int, default=8192)  # Reduced from 65536
    parser.add_argument('--teacher_temp', type=float, default=0.04)
    parser.add_argument('--student_temp', type=float, default=0.1)
    
    # Subgraph sampling (REDUCED DEFAULTS)
    parser.add_argument('--sampling_strategy', type=str, default='spatial')
    parser.add_argument('--subgraph_size', type=int, default=500)  # Reduced from 1000
    parser.add_argument('--subgraphs_per_graph', type=int, default=2)  # Reduced from 4
    parser.add_argument('--batch_size', type=int, default=2)  # Reduced from 8
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    # Memory optimization
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation', type=int, default=2,
                       help='Gradient accumulation steps')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*80)
    print("MEMORY-OPTIMIZED DINO TRAINING")
    print("="*80)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Find graph files
    data_dir = Path(args.data_dir)
    graph_files = list(data_dir.glob('*/nucleus_graph.pkl'))
    print(f"\nFound {len(graph_files)} graph files")
    
    if len(graph_files) == 0:
        print("Error: No graph files found!")
        return
    
    # Load first graph to get dimensions
    with open(graph_files[0], 'rb') as f:
        sample_graph = pickle.load(f)
    
    node_feat_dim = sample_graph['node_features'].shape[1]
    print(f"Node feature dimension: {node_feat_dim}")
    
    # Create model
    print(f"\nModel configuration:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  DINO out dim: {args.dino_out_dim}")
    
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
    teacher_head.load_state_dict(student_head.state_dict())
    for param in teacher_head.parameters():
        param.requires_grad = False
    
    # Create DINO loss
    dino_loss = DINOLoss(
        out_dim=args.dino_out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp
    )
    
    # Create sampler
    print(f"\nSampling configuration:")
    print(f"  Strategy: {args.sampling_strategy}")
    print(f"  Subgraph size: {args.subgraph_size}")
    print(f"  Subgraphs per graph: {args.subgraphs_per_graph}")
    print(f"  Batch size: {args.batch_size}")
    
    sampler = SubgraphSampler(
        strategy=args.sampling_strategy,
        subgraph_size=args.subgraph_size,
        overlap=0.2,
        min_nodes=100,
        max_nodes=1000
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
    trainer = MemoryEfficientDINOTrainer(
        model=model,
        student_head=student_head,
        teacher_head=teacher_head,
        dino_loss=dino_loss,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        log_dir=str(Path(args.output_dir) / 'logs'),
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for epoch in range(args.epochs):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  LR: {metrics['lr']:.6f}")
            print(f"  Momentum: {metrics['momentum']:.4f}")
            
            if torch.cuda.is_available():
                print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
                trainer.save_checkpoint(str(checkpoint_path), epoch, metrics)
        
        print("\n" + "="*80)
        print("Training complete!")
        print(f"Checkpoints saved to: {output_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        checkpoint_path = output_dir / 'checkpoint_interrupted.pt'
        trainer.save_checkpoint(str(checkpoint_path), epoch, metrics)
        print(f"Saved interrupted checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()