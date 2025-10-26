# Spatial Embedding Visualization for Nucleus Graphs

This guide explains how to create spatial embedding maps that mimic tissue shape, similar to your tile-based approach but for graphs.

## Concept Overview

### Tile-Based Approach (Images)
```
Image Tile → CNN → Embedding → Cluster → Plot at tile coordinates
```

### Graph-Based Approach (Two Options)

**Option 1: Subgraph Patches (Recommended - Direct Analog)**
```
Spatial Patch (subgraph) → Graph Transformer → Embedding → Cluster → Plot at patch center
```

**Option 2: Individual Nuclei**
```
Nucleus + k-hop neighbors → Graph Transformer → Embedding → Cluster → Plot at nucleus centroid
```

## How It Works

### Subgraph Patch Method

Think of it as **"graph tiles"**:

1. **Divide tissue into spatial patches** (e.g., 1000μm × 1000μm)
2. **Extract subgraph** within each patch (all nuclei + their connections)
3. **Embed each subgraph** using trained DINO model
4. **Cluster embeddings** to find tissue patterns
5. **Plot at patch centers** with colors by cluster

This creates a spatial map where:
- Similar tissue regions get similar colors
- Spatial patterns emerge (e.g., different layers, structures)
- You can see tissue architecture

### Visual Example

```
Original Tissue:
┌─────────────────────────┐
│ ███  ███  ███  ███  ███ │  <- Dense nuclei (cortex)
│ ███  ███  ███  ███  ███ │
│  ·    ·    ·    ·    ·  │  <- Sparse nuclei (medulla)
│    ·    ·    ·    ·    ·│
│ ███  ███  ███  ███  ███ │  <- Dense nuclei (follicles)
└─────────────────────────┘

After Clustering:
┌─────────────────────────┐
│ 🔴 🔴 🔴 🔴 🔴 🔴 🔴 🔴 │  <- Cluster 1 (cortex)
│ 🔴 🔴 🔴 🔴 🔴 🔴 🔴 🔴 │
│ 🔵 🔵 🔵 🔵 🔵 🔵 🔵 🔵 │  <- Cluster 2 (medulla)
│ 🔵 🔵 🔵 🔵 🔵 🔵 🔵 🔵 │
│ 🟢 🟢 🟢 🟢 🟢 🟢 🟢 🟢 │  <- Cluster 3 (follicles)
└─────────────────────────┘
```

## Usage

### 1. Single Slide Visualization

```bash
python visualize_spatial_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_file outputs/SLIDE_001/nucleus_graph.pkl \
    --output_dir visualizations/ \
    --mode subgraph \
    --patch_size 1000.0 \
    --overlap 0.5 \
    --viz_method cluster \
    --n_clusters 10
```

**Parameters:**
- `--mode subgraph`: Use spatial patches (like tiles)
- `--patch_size 1000.0`: 1000 micron patches
- `--overlap 0.5`: 50% overlap between patches (smoother maps)
- `--viz_method cluster`: Color by k-means clusters
- `--n_clusters 10`: Find 10 tissue types/patterns

**Output:**
- `SLIDE_001_embedding_map.png`: Side-by-side spatial map + UMAP
- `SLIDE_001_embedding_data.npz`: Raw embeddings + coordinates

### 2. Batch Process All Slides

```bash
python batch_visualize_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_dir outputs/ \
    --output_dir visualizations/ \
    --mode subgraph \
    --patch_size 1000.0 \
    --viz_method cluster \
    --n_clusters 10
```

Processes all slides in `outputs/` directory.

### 3. Different Visualization Methods

#### Method 1: Clustering (Default)
```bash
--viz_method cluster --n_clusters 10
```
- Clusters embeddings into discrete tissue types
- Best for identifying distinct regions
- Output: Categorical color map

#### Method 2: PCA
```bash
--viz_method pca
```
- Projects embeddings to first 3 principal components
- Shows continuous gradients across tissue
- Good for understanding major sources of variation

#### Method 3: UMAP
```bash
--viz_method umap
```
- Non-linear dimensionality reduction
- Preserves local + global structure
- Best for complex, non-linear patterns

## Interpreting Results

### Spatial Map (Left Panel)

The left panel shows your tissue with colors representing learned patterns:

**What to look for:**
- **Distinct regions**: Do clusters align with known anatomy?
- **Gradients**: Smooth transitions between tissue types
- **Boundaries**: Sharp changes at architectural boundaries
- **Repeated patterns**: Similar structures across the slide

**Example interpretations (ovary):**
- 🔴 **Red cluster**: Cortex (dense granulosa cells)
- 🔵 **Blue cluster**: Medulla (loose connective tissue)
- 🟢 **Green cluster**: Follicles (oocytes + surrounding cells)
- 🟡 **Yellow cluster**: Corpus luteum (if present)
- 🟣 **Purple cluster**: Blood vessels

### UMAP/Embedding Space (Right Panel)

The right panel shows how embeddings cluster in high-dimensional space:

**What to look for:**
- **Separated clusters**: Distinct tissue types
- **Connected clusters**: Related tissue types
- **Outliers**: Unusual or rare structures
- **Density**: Common vs rare patterns

## Advanced Usage

### 1. Node-Level Visualization (Higher Resolution)

Instead of patches, visualize each nucleus individually:

```bash
python visualize_spatial_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_file outputs/SLIDE_001/nucleus_graph.pkl \
    --mode node \
    --viz_method cluster \
    --n_clusters 15
```

**Pros:**
- Higher spatial resolution
- See individual cell patterns

**Cons:**
- Much slower (processes all nuclei)
- More noisy (single cells vs neighborhoods)

### 2. Adjust Patch Size

Smaller patches → more detail:
```bash
--patch_size 500.0  # 500μm patches
```

Larger patches → smoother, more context:
```bash
--patch_size 2000.0  # 2000μm patches
```

### 3. Overlap for Smoother Maps

Higher overlap → smoother transitions:
```bash
--overlap 0.7  # 70% overlap
```

Lower overlap → faster, but blockier:
```bash
--overlap 0.3  # 30% overlap
```

### 4. More Clusters for Finer Detail

```bash
--n_clusters 20  # Find 20 patterns (more granular)
```

```bash
--n_clusters 5   # Find 5 patterns (broader categories)
```

## Programmatic Access

Load and analyze the raw data:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load embedding data
data = np.load('visualizations/SLIDE_001_embedding_data.npz')
embeddings = data['embeddings']    # (N_patches, 1280)
coordinates = data['coordinates']  # (N_patches, 2)

# Cluster
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Custom visualization
plt.figure(figsize=(12, 10))
plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=clusters,
    cmap='tab20',
    s=100,
    alpha=0.8
)
plt.axis('equal')
plt.title('Custom Spatial Map')
plt.savefig('custom_map.png', dpi=300)
```

### Compare Multiple Slides

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load embeddings from multiple slides
slides = ['SLIDE_001', 'SLIDE_002', 'SLIDE_003']
all_embeddings = []
all_labels = []

for slide in slides:
    data = np.load(f'visualizations/{slide}_embedding_data.npz')
    all_embeddings.append(data['embeddings'])
    all_labels.extend([slide] * len(data['embeddings']))

# Combine
embeddings = np.vstack(all_embeddings)

# PCA across all slides
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
for slide in slides:
    mask = np.array(all_labels) == slide
    ax.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=slide,
        alpha=0.6,
        s=50
    )
ax.legend()
ax.set_title('Embedding Space Across Slides')
plt.savefig('multi_slide_comparison.png', dpi=300)
```

### Find Similar Regions Across Slides

```python
from sklearn.metrics.pairwise import cosine_similarity

# Load reference patch from SLIDE_001
ref_data = np.load('visualizations/SLIDE_001_embedding_data.npz')
ref_patch_idx = 50  # Choose a patch
ref_embedding = ref_data['embeddings'][ref_patch_idx:ref_patch_idx+1]

# Find similar patches in SLIDE_002
query_data = np.load('visualizations/SLIDE_002_embedding_data.npz')
query_embeddings = query_data['embeddings']

# Compute similarities
similarities = cosine_similarity(ref_embedding, query_embeddings)[0]

# Get most similar patches
top_k = 5
similar_indices = np.argsort(similarities)[-top_k:][::-1]

print(f"Most similar patches to SLIDE_001 patch {ref_patch_idx}:")
for idx in similar_indices:
    coord = query_data['coordinates'][idx]
    print(f"  Patch at ({coord[0]:.1f}, {coord[1]:.1f}): "
          f"similarity = {similarities[idx]:.3f}")
```

## Tips for Best Results

### 1. Choose Appropriate Patch Size

**For ovary tissue:**
- Small follicles: 500-800μm patches
- Large follicles: 1000-1500μm patches
- Whole structure analysis: 2000μm+ patches

### 2. Balance Speed vs Resolution

**Fast (coarse):**
```bash
--patch_size 2000 --overlap 0.3 --mode subgraph
```

**Slow (fine):**
```bash
--patch_size 500 --overlap 0.7 --mode node
```

### 3. Validate Clusters

After clustering, validate by:
1. Checking if clusters align with known anatomy
2. Looking at feature distributions per cluster
3. Comparing to pathologist annotations

### 4. Iterate on Number of Clusters

Try different values:
```bash
for k in 5 10 15 20; do
    python visualize_spatial_embeddings.py \
        --n_clusters $k \
        --output_dir visualizations/k${k}/
done
```

Compare results to find optimal granularity.

## Troubleshooting

### Issue: Uniform colors across tissue
**Problem**: Model hasn't learned meaningful patterns
**Solution**: Train longer, check if model converged

### Issue: Too many small clusters
**Problem**: Too sensitive to noise
**Solution**: 
- Increase patch size
- Reduce n_clusters
- Use more overlap

### Issue: Clusters don't match anatomy
**Problem**: Model learning technical artifacts
**Solution**:
- Check data quality
- Retrain with better augmentations
- Try different visualization methods

### Issue: Out of memory
**Problem**: Too many patches
**Solution**:
- Increase patch size
- Reduce overlap
- Process in batches

## Integration with Existing Pipeline

### Complete Workflow

```bash
# 1. Process WSI to graphs (already done)
sbatch run_array.sh

# 2. Train DINO model
sbatch run_dino_training.sh

# 3. Generate spatial embedding maps
python batch_visualize_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_dir outputs/ \
    --output_dir visualizations/

# 4. Generate slide-level embeddings
python infer_embeddings.py \
    --checkpoint dino_checkpoints/checkpoint_epoch_100.pt \
    --graph_dir outputs/ \
    --output_file embeddings/slide_embeddings.npz
```

Now you have:
- **Spatial maps**: Visualize tissue architecture
- **Slide embeddings**: For classification/clustering
- **Raw embedding data**: For custom analysis

## Next Steps

1. **Validate**: Compare clusters to pathologist annotations
2. **Analyze**: Correlate with clinical outcomes
3. **Refine**: Adjust parameters based on biological knowledge
4. **Scale**: Process entire cohort
5. **Discover**: Find novel tissue patterns