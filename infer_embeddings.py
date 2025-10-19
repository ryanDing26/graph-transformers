"""
Generate embeddings for new WSI slides using trained DINO model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm

import sys
sys.path.append('models')
from student_teacher import StudentTeacherGraphTransformer
from subgraph_sampler import SubgraphSampler
from train_dino_subgraphs import InferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for WSI graphs'
    )
    
    # Input
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--graph_dir', type=str, required=True,
                       help='Directory containing graph .pkl files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file for embeddings (.npz or .h5)')
    
    # Inference settings
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of subgraphs to sample per graph')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'max', 'concat'],
                       help='How to aggregate subgraph embeddings')
    parser.add_argument('--sampling_strategy', type=str, default='spatial',
                       choices=['spatial', 'khop', 'random_walk'])
    parser.add_argument('--subgraph_size', type=int, default=1000)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of graphs to process in parallel')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING WSI EMBEDDINGS")
    print("="*80)
    
    # Find graph files
    graph_dir = Path(args.graph_dir)
    graph_files = sorted(list(graph_dir.glob('*/nucleus_graph.pkl')))
    
    if len(graph_files) == 0:
        print(f"Error: No graph files found in {graph_dir}")
        return
    
    print(f"\nFound {len(graph_files)} graphs to process")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load first graph to get dimensions
    with open(graph_files[0], 'rb') as f:
        sample_graph = pickle.load(f)
    node_feat_dim = sample_graph['node_features'].shape[1]
    
    # Reconstruct model architecture from checkpoint
    # You might want to save these params in the checkpoint
    model = StudentTeacherGraphTransformer(
        node_feat_dim=node_feat_dim,
        hidden_dim=256,  # Should match training
        num_heads=8,
        num_layers=4,
        output_dim=1280,
        use_edge_features=False,
        ema_momentum=0.996,
        use_teacher_for_inference=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    
    # Create sampler
    sampler = SubgraphSampler(
        strategy=args.sampling_strategy,
        subgraph_size=args.subgraph_size,
        overlap=0.2,
        min_nodes=200,
        max_nodes=2000
    )
    
    # Create inference engine
    inference = InferenceEngine(
        model=model,
        sampler=sampler,
        device=args.device,
        use_teacher=True
    )
    
    # Process all graphs
    print(f"\nProcessing {len(graph_files)} graphs...")
    print(f"  Samples per graph: {args.num_samples}")
    print(f"  Aggregation: {args.aggregation}")
    print("")
    
    embeddings = []
    slide_names = []
    metadata = []
    
    for graph_file in tqdm(graph_files, desc="Embedding graphs"):
        # Load graph
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Generate embedding
        embedding = inference.embed_graph(
            graph_data,
            num_samples=args.num_samples,
            aggregation=args.aggregation
        )
        
        # Store results
        slide_name = graph_file.parent.name
        embeddings.append(embedding)
        slide_names.append(slide_name)
        
        # Store metadata
        metadata.append({
            'slide_name': slide_name,
            'num_nodes': graph_data['num_nodes'],
            'num_edges': len(graph_data['edge_index']),
            'embedding_dim': len(embedding)
        })
    
    embeddings = np.array(embeddings)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.npz':
        # Save as NumPy archive
        np.savez(
            output_path,
            embeddings=embeddings,
            slide_names=np.array(slide_names),
            metadata=metadata
        )
    elif output_path.suffix == '.h5':
        # Save as HDF5
        import h5py
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('slide_names', data=np.array(slide_names, dtype='S'))
            
            # Save metadata as attributes
            for i, meta in enumerate(metadata):
                grp = f.create_group(f'metadata/{slide_names[i]}')
                for key, val in meta.items():
                    grp.attrs[key] = val
    else:
        # Save as CSV (embeddings) + JSON (metadata)
        df = pd.DataFrame(embeddings)
        df['slide_name'] = slide_names
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        import json
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved embeddings to: {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Summary statistics
    print("\nEmbedding Statistics:")
    print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"  Std norm: {np.linalg.norm(embeddings, axis=1).std():.4f}")
    
    print("\n" + "="*80)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()