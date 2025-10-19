# DINO Training for Large Nucleus Graphs

This guide explains how to train DINO models on large nucleus graphs (10K+ nodes, 100K+ edges) using subgraph sampling.

## Pipeline Overview

```
WSI Slides (.svs)
    ↓
[1] Nucleus Segmentation & Feature Extraction
    → nucleus_graph_pipeline.py
    → Outputs: nucleus_features.csv, nucleus_graph.pkl
    ↓
[2] Subgraph Sampling & DINO Training
    → train_dino_subgraphs.py
    → Outputs: trained checkpoints
    ↓
[3] Inference & Embedding Generation
    → infer_embeddings.py
    → Outputs: slide_embeddings.npz
```

## 1. Process WSI Slides (Already Done)

You've already generated nucleus graphs using `batch_process_wsi.py`. Each slide should have:
- `outputs/SLIDE_NAME/nucleus_features.csv` - Node features
- `outputs/SLIDE_NAME/nucleus_graph.pkl` - Graph structure

Your graphs have:
- **~10,000-50,000 nodes** (nuclei)
- **~100,000-500,000 edges** (spatial connections)
- **~100-200 features per node** (morphology, intensity, texture, aging markers)

## 2. Training Configuration

### Subgraph Sampling Strategies

**Spatial Sampling (Recommended for Nucleus Graphs)**
- Samples square spatial regions based on nucleus coordinates
- Preserves local tissue architecture
- Best for histology where spatial context matters
```bash
--sampling_strategy spatial --subgraph_size 1000
```

**K-hop Sampling**
- Samples k-hop neighborhoods around random seed nodes
- Preserves graph connectivity structure
- Good for general graph learning
```bash
--sampling_strategy khop --subgraph_size 1000
```

**Random Walk Sampling**
- Uses random walks to sample connected subgraphs
- Creates cohesive local neighborhoods
- Good for community detection
```bash
--sampling_strategy random_walk --subgraph_size 1000
```

### Hyperparameters

**Model Architecture:**
- `--hidden_dim 256` - Transformer hidden dimension
- `--num_heads 8` - Multi-head attention heads
- `--num_layers 4` - Number of transformer layers
- `--output_dim 1280` - Final embedding dimension (matches DINOv2)

**DINO Parameters:**
- `--dino_out_dim 65536` - DINO projection dimension (high-dim space)
- `--teacher_temp 0.04` - Teacher temperature (sharpening)
- `--student_temp 0.1` - Student temperature
- `--warmup_epochs 10` - Learning rate warmup

**Subgraph Sampling:**
- `--subgraph_size 1000` - Target nodes per subgraph
- `--subgraphs_per_graph 4` - Samples per graph per epoch
- `--batch_size 8` - Subgraphs per batch

### Memory Considerations

For a **1000-node subgraph**:
- Node features: `1000 × 150 features × 4 bytes = 600 KB`
- Attention matrix: `1000 × 1000 × 4 bytes = 4 MB`
- Total per subgraph: **~20-50 MB**

With `batch_size=8`: **~400 MB** per batch
→ Easily fits on any GPU with 8GB+ VRAM

For **2000-node subgraphs**, double these numbers.

## 3. Training Commands

### Single GPU Training

```bash
python train_dino_subgraphs.py \
    --data_dir /shares/sinha/rding/graph-transformers/outputs \
    --output_dir /shares/sinha/rding/graph-transformers/dino_checkpoints \
    --sampling_strategy spatial \
    --subgraph_size 1000 \
    --subgraphs_per_graph 4 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda
```

### Multi-GPU Training (Data Parallel)

Create `train_dino_ddp.py` wrapper:
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Train as usual
```

### SLURM Array Job (Parallel Data Processing)

If you want to train on subsets of data in parallel:

```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

python train_dino_subgraphs.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --epochs 100 \
    --checkpoint_dir checkpoints/array_${SLURM_ARRAY_TASK_ID}
```

## 4. Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/dino/logs --port 6006
```

Metrics to watch:
- **Loss**: Should decrease and stabilize
- **Learning Rate**: Warmup → cosine decay
- **EMA Momentum**: Should increase 0.996 → 0.999

### Expected Training Time

For **250 WSI slides** × **4 subgraphs/slide** × **100 epochs**:
- Total samples: `250 × 4 × 100 = 100,000`
- Batch size: `8`
- Batches per epoch: `1,000`
- Time per batch: `~0.5s`

**Estimated time: 14-16 hours on single V100/A100**

## 5. Inference & Embedding Generation

After training, generate embeddings for all slides:

```bash
python infer_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_dir outputs/ \
    --output_file embeddings/slide_embeddings.npz \
    --num_samples 10 \
    --aggregation mean
```

### Aggregation Strategies

**Mean Pooling (Recommended)**
- Average embeddings from multiple subgraph samples
- Robust to sampling variance
- Output: `1280-dim vector` per slide

**Max Pooling**
- Takes max across subgraph samples
- Emphasizes dominant features
- Output: `1280-dim vector` per slide

**Concatenation**
- Concatenates all subgraph embeddings
- Preserves all information
- Output: `(1280 × num_samples)-dim vector` (e.g., 12,800-dim)

## 6. Using the Embeddings

### Load Embeddings

```python
import numpy as np

# Load embeddings
data = np.load('embeddings/slide_embeddings.npz', allow_pickle=True)
embeddings = data['embeddings']  # (N_slides, 1280)
slide_names = data['slide_names']  # (N_slides,)
metadata = data['metadata']  # List of dicts

print(f"Loaded embeddings: {embeddings.shape}")
```

### Downstream Tasks

**1. Classification (Tissue Type, Disease State)**
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(embeddings_train, labels_train)
predictions = clf.predict(embeddings_test)
```

**2. Clustering (Discover Tissue Subtypes)**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)
```

**3. Similarity Search (Find Similar Slides)**
```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)
most_similar = np.argsort(similarities[query_idx])[::-1]
```

**4. Dimensionality Reduction (Visualization)**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.savefig('embedding_visualization.png')
```

## 7. Optimization Tips

### If Training is Too Slow

1. **Increase batch size** (if GPU memory allows):
   ```bash
   --batch_size 16  # Instead of 8
   ```

2. **Reduce number of transformer layers**:
   ```bash
   --num_layers 3  # Instead of 4
   ```

3. **Use larger subgraphs, fewer samples**:
   ```bash
   --subgraph_size 1500 --subgraphs_per_graph 2
   ```

### If Running Out of Memory

1. **Reduce subgraph size**:
   ```bash
   --subgraph_size 500  # Instead of 1000
   ```

2. **Reduce batch size**:
   ```bash
   --batch_size 4  # Instead of 8
   ```

3. **Enable gradient checkpointing** (add to model):
   ```python
   from torch.utils.checkpoint import checkpoint
   # In transformer layer forward:
   x = checkpoint(self.attention, x, adj_mask)
   ```

### If Loss Not Decreasing

1. **Check data quality**: Ensure graphs are valid
2. **Adjust temperatures**: Try `--teacher_temp 0.07 --student_temp 0.1`
3. **Increase warmup**: `--warmup_epochs 20`
4. **Check augmentations**: May be too aggressive

## 8. Advanced: Hierarchical Approach

For extremely large graphs (100K+ nodes), use a hierarchical approach:

1. **Level 1**: Sample 10-20 subgraphs per slide
2. **Level 2**: Process each subgraph → 1280-dim embedding
3. **Level 3**: Aggregate with attention pooling

```python
class HierarchicalEncoder(nn.Module):
    def __init__(self):
        self.subgraph_encoder = GraphTransformer(...)
        self.aggregator = nn.TransformerEncoder(...)
    
    def forward(self, subgraphs):
        # Encode each subgraph
        subgraph_embeds = [self.subgraph_encoder(sg) for sg in subgraphs]
        
        # Aggregate with transformer
        slide_embed = self.aggregator(torch.stack(subgraph_embeds))
        return slide_embed
```

## 9. Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce `--batch_size` or `--subgraph_size`

### Issue: "No convergence"
**Solution**: 
- Check learning rate (try `--lr 5e-5`)
- Increase warmup epochs
- Verify data quality

### Issue: "Too slow"
**Solution**:
- Use spatial sampling (fastest)
- Increase batch size
- Use mixed precision training (add `torch.cuda.amp`)

### Issue: "Poor downstream performance"
**Solution**:
- Train longer (150-200 epochs)
- Increase model capacity (`--hidden_dim 512`)
- Try different sampling strategies
- Add more augmentations

## 10. Next Steps

After training:
1. **Validate embeddings** on held-out test set
2. **Fine-tune** for specific downstream tasks
3. **Analyze learned representations** (t-SNE, attention maps)
4. **Compare** to other methods (ResNet features, etc.)

## Resources

- **DINO paper**: https://arxiv.org/abs/2104.14294
- **DINOv2 paper**: https://arxiv.org/abs/2304.07193
- **Graph Transformers**: https://arxiv.org/abs/2012.09699