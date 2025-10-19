"""
Batch process multiple slides to create spatial embedding maps.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import sys
sys.path.append('models')
from student_teacher import StudentTeacherGraphTransformer
from visualize_spatial_embeddings import (
    SpatialEmbeddingMapper,
    visualize_spatial_embeddings
)


def process_single_slide(args_tuple):
    """Process a single slide (for parallel execution)."""
    graph_file, checkpoint_path, output_dir, mode, patch_size, overlap, viz_method, n_clusters, device = args_tuple
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load graph
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Reconstruct model
        node_feat_dim = graph_data['node_features'].shape[1]
        model = StudentTeacherGraphTransformer(
            node_feat_dim=node_feat_dim,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            output_dim=1280,
            use_edge_features=False,
            use_teacher_for_inference=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create mapper
        mapper = SpatialEmbeddingMapper(
            model=model,
            device=device,
            use_teacher=True
        )
        
        # Generate embedding map
        if mode == 'subgraph':
            embeddings, coordinates, node_counts = mapper.create_subgraph_embedding_map(
                graph_data=graph_data,
                patch_size=patch_size,
                overlap=overlap,
                min_nodes=50
            )
        else:
            embeddings, coordinates = mapper.create_node_embedding_map(
                graph_data=graph_data,
                batch_size=100
            )
        
        # Visualize
        slide_name = Path(graph_file).parent.name
        output_path = Path(output_dir) / f'{slide_name}_embedding_map.png'
        
        visualize_spatial_embeddings(
            embeddings=embeddings,
            coordinates=coordinates,
            output_path=str(output_path),
            method=viz_method,
            n_clusters=n_clusters
        )
        
        # Save raw data
        data_path = Path(output_dir) / f'{slide_name}_embedding_data.npz'
        np.savez(
            data_path,
            embeddings=embeddings,
            coordinates=coordinates,
            slide_name=slide_name
        )
        
        return (slide_name, True, len(embeddings))
        
    except Exception as e:
        slide_name = Path(graph_file).parent.name
        print(f"Error processing {slide_name}: {str(e)}")
        return (slide_name, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Batch create spatial embedding maps for all slides'
    )
    
    # Input
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--graph_dir', type=str, required=True,
                       help='Directory containing graph .pkl files')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for plots')
    
    # Mapping method
    parser.add_argument('--mode', type=str, default='subgraph',
                       choices=['subgraph', 'node'],
                       help='Use subgraph patches or individual nodes')
    
    # Subgraph parameters
    parser.add_argument('--patch_size', type=float, default=1000.0,
                       help='Patch size in microns')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap between patches')
    
    # Visualization
    parser.add_argument('--viz_method', type=str, default='cluster',
                       choices=['cluster', 'pca', 'umap'],
                       help='Visualization method')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers (use 1 for GPU)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH SPATIAL EMBEDDING MAP GENERATION")
    print("="*80)
    
    # Find all graph files
    graph_dir = Path(args.graph_dir)
    graph_files = sorted(list(graph_dir.glob('*/nucleus_graph.pkl')))
    
    print(f"\nFound {len(graph_files)} graphs to process")
    
    if len(graph_files) == 0:
        print(f"Error: No graph files found in {graph_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing settings:")
    print(f"  Mode: {args.mode}")
    print(f"  Patch size: {args.patch_size} microns")
    print(f"  Overlap: {args.overlap}")
    print(f"  Visualization: {args.viz_method}")
    print(f"  Clusters: {args.n_clusters}")
    print(f"  Workers: {args.num_workers}")
    print("")
    
    # Prepare arguments for each slide
    task_args = [
        (
            str(graph_file),
            args.checkpoint,
            str(output_dir),
            args.mode,
            args.patch_size,
            args.overlap,
            args.viz_method,
            args.n_clusters,
            args.device
        )
        for graph_file in graph_files
    ]
    
    # Process slides
    results = []
    
    if args.num_workers == 1:
        # Sequential processing (better for GPU)
        for task in tqdm(task_args, desc="Processing slides"):
            result = process_single_slide(task)
            results.append(result)
    else:
        # Parallel processing (use CPU only)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_slide, task_args),
                total=len(task_args),
                desc="Processing slides"
            ))
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\nTotal slides: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_patches = sum(r[2] for r in successful)
        print(f"\nTotal patches generated: {total_patches}")
        print(f"Average patches per slide: {total_patches / len(successful):.1f}")
    
    if failed:
        print(f"\nFailed slides:")
        for slide_name, _, error in failed:
            print(f"  - {slide_name}: {error}")
    
    print(f"\n✓ Output saved to: {output_dir}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()