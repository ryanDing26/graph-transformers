#!/usr/bin/env python3
"""
batch_process_wsi.py

Process all .svs files in a directory with the nucleus graph pipeline
Can be run standalone or as part of SLURM array job
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import traceback

# Import the pipeline
from nucleus_graph_pipeline import LazySlideNucleusExtractor, GlobalGraphBuilder


def setup_logging(output_dir: Path, log_name: str = "processing"):
    """Setup logging to file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def find_svs_files(input_dir: Path) -> list:
    """Find all .svs files in directory"""
    svs_files = sorted(list(input_dir.glob("*.svs")))
    logging.info(f"Found {len(svs_files)} .svs files in {input_dir}")
    return svs_files


def process_single_wsi(
    wsi_path: Path,
    output_dir: Path,
    tile_size: int = 512,
    mpp: float = 0.5,
    batch_size: int = 32,
    k_neighbors: int = 10,
    max_distance: float = 500,
    use_pe: bool = False,
    skip_existing: bool = True
) -> bool:
    """
    Process a single WSI file
    
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Create output subdirectory for this slide
    slide_name = wsi_path.stem
    slide_output_dir = output_dir / slide_name
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    features_path = slide_output_dir / 'nucleus_features.csv'
    graph_path = slide_output_dir / 'nucleus_graph.pkl'
    
    if skip_existing and features_path.exists() and graph_path.exists():
        logger.info(f"✓ Skipping {slide_name} (already processed)")
        return True
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {slide_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Step 1: Extract nucleus features
        logger.info("\n[1/2] Extracting nucleus features...")
        extractor = LazySlideNucleusExtractor(
            patch_size=128,
            min_nucleus_area=20,
            max_nucleus_area=10000
        )
        
        features_df = extractor.process_wsi(
            wsi_path=str(wsi_path),
            output_path=features_path,
            tile_size=tile_size,
            mpp=mpp,
            batch_size=batch_size
        )
        
        if len(features_df) == 0:
            logger.warning(f"⚠ No nuclei found in {slide_name}")
            return False
        
        # Step 2: Build graph
        logger.info("\n[2/2] Building global graph...")
        graph_builder = GlobalGraphBuilder(
            k_neighbors=k_neighbors,
            max_edge_distance=max_distance,
            use_laplacian_pe=use_pe,
            n_pe_dims=16
        )
        
        graph_data = graph_builder.build_graph(
            features_df=features_df,
            output_path=graph_path
        )
        
        # Save summary
        summary_path = slide_output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Slide: {slide_name}\n")
            f.write(f"Processed: {datetime.now()}\n")
            f.write(f"Nuclei: {graph_data['num_nodes']}\n")
            f.write(f"Edges: {len(graph_data['edge_index'])}\n")
            f.write(f"Features: {graph_data['node_features'].shape[1]}\n")
        
        logger.info(f"\n✓ Successfully processed {slide_name}")
        logger.info(f"  - Nuclei: {graph_data['num_nodes']}")
        logger.info(f"  - Edges: {len(graph_data['edge_index'])}")
        logger.info(f"  - Features: {graph_data['node_features'].shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Failed to process {slide_name}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save error info
        error_path = slide_output_dir / 'error.txt'
        with open(error_path, 'w') as f:
            f.write(f"Error processing {slide_name}\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    tile_size: int = 512,
    mpp: float = 0.5,
    batch_size: int = 32,
    k_neighbors: int = 10,
    max_distance: float = 500,
    use_pe: bool = False,
    skip_existing: bool = True,
    start_idx: int = 0,
    end_idx: int = None
):
    """
    Process all .svs files in a directory
    
    Args:
        start_idx: Start index for array jobs (inclusive)
        end_idx: End index for array jobs (exclusive)
    """
    logger = setup_logging(output_dir, "batch_processing")
    
    logger.info("\n" + "="*80)
    logger.info("BATCH WSI PROCESSING - NUCLEUS GRAPH PIPELINE")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"MPP: {mpp}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"K neighbors: {k_neighbors}")
    logger.info(f"Max distance: {max_distance}")
    logger.info(f"Use Laplacian PE: {use_pe}")
    logger.info(f"Skip existing: {skip_existing}")
    
    # Find all .svs files
    svs_files = find_svs_files(input_dir)
    
    if len(svs_files) == 0:
        logger.error("No .svs files found!")
        return
    
    # Handle array job indexing
    if end_idx is None:
        end_idx = len(svs_files)
    
    svs_files = svs_files[start_idx:end_idx]
    logger.info(f"\nProcessing files {start_idx} to {end_idx-1} ({len(svs_files)} files)")
    
    # Process each file
    results = []
    for i, svs_path in enumerate(svs_files, start=start_idx):
        logger.info(f"\n[{i+1}/{end_idx}] {svs_path.name}")
        
        success = process_single_wsi(
            wsi_path=svs_path,
            output_dir=output_dir,
            tile_size=tile_size,
            mpp=mpp,
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            max_distance=max_distance,
            use_pe=use_pe,
            skip_existing=skip_existing
        )
        
        results.append({
            'file': svs_path.name,
            'success': success
        })
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.info("\nFailed files:")
        for r in results:
            if not r['success']:
                logger.info(f"  - {r['file']}")
    
    # Save summary
    summary_path = output_dir / f'batch_summary_{start_idx}_{end_idx}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Batch Processing Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Processed: {datetime.now()}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"File range: {start_idx} to {end_idx-1}\n\n")
        f.write(f"Total files: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        if failed > 0:
            f.write("Failed files:\n")
            for r in results:
                if not r['success']:
                    f.write(f"  - {r['file']}\n")
    
    logger.info(f"\nBatch summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process WSI files with nucleus graph pipeline'
    )
    
    # Required arguments
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing .svs files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for results'
    )
    
    # Processing parameters
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--mpp', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--max_distance', type=float, default=500)
    parser.add_argument('--use_pe', action='store_true', help='Use Laplacian PE')
    parser.add_argument('--no_skip', action='store_true', help='Reprocess existing files')
    
    # Array job support
    parser.add_argument(
        '--array_idx',
        type=int,
        default=None,
        help='SLURM array task ID (for parallel processing)'
    )
    parser.add_argument(
        '--files_per_job',
        type=int,
        default=1,
        help='Number of files to process per array job'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle array job indexing
    if args.array_idx is not None:
        # SLURM array job - process a subset of files
        start_idx = args.array_idx * args.files_per_job
        end_idx = start_idx + args.files_per_job
    else:
        # Process all files
        start_idx = 0
        end_idx = None
    
    # Run processing
    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=args.tile_size,
        mpp=args.mpp,
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        max_distance=args.max_distance,
        use_pe=args.use_pe,
        skip_existing=not args.no_skip,
        start_idx=start_idx,
        end_idx=end_idx
    )


if __name__ == '__main__':
    main()