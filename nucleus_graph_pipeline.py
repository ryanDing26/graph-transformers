"""
Phase 1: Nucleus-Only Graph Construction for Aging Analysis
Uses LazySlide + InstanSeg for nucleus segmentation
Builds global graph with shape-invariant features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# LazySlide and WSI
import lazyslide as zs
from wsidata import open_wsi

# Image processing
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage
import mahotas

# Graph construction
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point, Polygon
from shapely.affinity import translate
import geopandas as gpd


@dataclass
class NucleusFeatures:
    """Container for nucleus-level features (no cell/cytoplasm)"""
    nucleus_id: int
    centroid_x: float
    centroid_y: float
    
    # Morphological features
    morphology: Dict[str, float] = field(default_factory=dict)
    
    # Intensity features
    intensity: Dict[str, float] = field(default_factory=dict)
    
    # Texture features
    texture: Dict[str, float] = field(default_factory=dict)
    
    # Aging-specific features (nucleus-only)
    aging_markers: Dict[str, float] = field(default_factory=dict)
    
    # Spatial context features (shape-invariant)
    spatial_context: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for DataFrame"""
        result = {
            'nucleus_id': self.nucleus_id,
            'centroid_x': self.centroid_x,
            'centroid_y': self.centroid_y,
        }
        result.update({f'morph_{k}': v for k, v in self.morphology.items()})
        result.update({f'intensity_{k}': v for k, v in self.intensity.items()})
        result.update({f'texture_{k}': v for k, v in self.texture.items()})
        result.update({f'aging_{k}': v for k, v in self.aging_markers.items()})
        result.update({f'spatial_{k}': v for k, v in self.spatial_context.items()})
        
        return result


class LazySlideNucleusExtractor:
    """
    Extract nucleus features from LazySlide/InstanSeg output
    Phase 1: Nucleus-only, no cell boundaries
    """
    
    def __init__(
        self,
        patch_size: int = 128,
        min_nucleus_area: int = 20,
        max_nucleus_area: int = 10000,
    ):
        """
        Initialize nucleus feature extractor
        
        Args:
            patch_size: Size of image patch to extract around each nucleus
            min_nucleus_area: Minimum nucleus area in pixels
            max_nucleus_area: Maximum nucleus area in pixels
        """
        self.patch_size = patch_size
        self.min_nucleus_area = min_nucleus_area
        self.max_nucleus_area = max_nucleus_area
    
    def process_wsi(
        self,
        wsi_path: str,
        output_path: Optional[str] = None,
        tile_size: int = 512,
        mpp: float = 0.5,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Complete pipeline: segmentation + feature extraction
        
        Args:
            wsi_path: Path to whole slide image
            output_path: Optional path to save features CSV
            tile_size: Tile size for processing
            mpp: Microns per pixel for processing
            batch_size: Batch size for InstanSeg
        
        Returns:
            DataFrame with all nucleus features
        """
        print(f"Processing WSI: {wsi_path}")
        print("=" * 60)
        
        # Step 1: Load WSI
        print("\n[1/4] Loading WSI...")
        wsi = open_wsi(wsi_path)
        
        # Step 2: Find and tile tissues
        print("[2/4] Finding tissues and tiling...")
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(
            wsi, 
            tile_size, 
            overlap=0.2, 
            background_fraction=0.95, 
            mpp=mpp
        )
        
        # Step 3: Segment nuclei with InstanSeg
        print("[3/4] Segmenting nuclei with InstanSeg...")
        zs.seg.cells(wsi, batch_size=batch_size)
        
        # Step 4: Extract features
        print("[4/4] Extracting features from nuclei...")
        features_df = self.extract_features_from_wsi(wsi)
        
        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved features to {output_path}")
        
        print(f"\n✓ Successfully extracted features from {len(features_df)} nuclei")
        print(f"✓ Feature dimensions: {features_df.shape}")
        
        return features_df
    
    def extract_features_from_wsi(self, wsi) -> pd.DataFrame:
        """
        Extract features from already-segmented WSI
        
        Args:
            wsi: LazySlide WSIData object with segmentation results
        
        Returns:
            DataFrame with nucleus features
        """
        # Get nucleus polygons from LazySlide
        # Format: GeoDataFrame with columns ['prob', 'geometry']
        if "cells" not in wsi.shapes:
            raise ValueError("No cell segmentation found. Run zs.seg.cells(wsi) first.")
        
        nuclei_gdf = wsi.shapes["cells"]
        print(f"Found {len(nuclei_gdf)} nuclei")
        print(f"Columns: {list(nuclei_gdf.columns)}")
        
        # Filter by area
        nuclei_gdf['area'] = nuclei_gdf.geometry.area
        nuclei_gdf = nuclei_gdf[
            (nuclei_gdf['area'] >= self.min_nucleus_area) &
            (nuclei_gdf['area'] <= self.max_nucleus_area)
        ].copy()
        print(f"After area filtering: {len(nuclei_gdf)} nuclei")
        
        if len(nuclei_gdf) == 0:
            print("Warning: No nuclei passed filtering!")
            return pd.DataFrame()
        
        # Extract centroids for spatial features later
        centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in nuclei_gdf.geometry
        ])
        
        # Extract features for each nucleus
        all_features = []
        
        for idx, (gdf_idx, row) in enumerate(nuclei_gdf.iterrows()):
            if idx % 100 == 0:
                print(f"  Processing nucleus {idx}/{len(nuclei_gdf)}")
            
            try:
                features = self.extract_nucleus_features(
                    wsi=wsi,
                    nucleus_polygon=row.geometry,
                    nucleus_id=idx,
                    centroid=centroids[idx]
                )
                all_features.append(features)
                
            except Exception as e:
                print(f"  Warning: Error processing nucleus {idx}: {str(e)}")
                continue
        
        if len(all_features) == 0:
            print("Warning: No features extracted successfully!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in all_features])
        
        # Add spatial context features (shape-invariant)
        print("Computing spatial context features...")
        df = self._add_spatial_features(df)
        
        return df
    
    def extract_nucleus_features(
        self,
        wsi,
        nucleus_polygon: Polygon,
        nucleus_id: int,
        centroid: np.ndarray
    ) -> NucleusFeatures:
        """
        Extract all features for a single nucleus
        
        Args:
            wsi: LazySlide WSIData object
            nucleus_polygon: Shapely Polygon for nucleus
            nucleus_id: Unique nucleus identifier
            centroid: Nucleus centroid coordinates (x, y)
        
        Returns:
            NucleusFeatures object
        """
        features = NucleusFeatures(
            nucleus_id=nucleus_id,
            centroid_x=centroid[0],
            centroid_y=centroid[1]
        )
        
        # Get image patch around nucleus
        half_patch = self.patch_size // 2
        cx, cy = int(centroid[0]), int(centroid[1])
        
        try:
            # Read region from WSI
            # Note: wsi is a SpatialData object, need to read the image
            # The actual image reading depends on wsidata implementation
            # Try to get the underlying image reader
            
            # Get the image at level 0 (highest resolution)
            location = (cx - half_patch, cy - half_patch)
            size = (self.patch_size, self.patch_size)
            
            # Read using wsidata's read_region method
            nucleus_image = wsi.read_region(
                x=cx-half_patch, y=cy-half_patch,
                width=self.patch_size, height=self.patch_size,
                level=0
            )
            
            # Convert to numpy array if needed
            if not isinstance(nucleus_image, np.ndarray):
                nucleus_image = np.array(nucleus_image)
            
            # Ensure RGB
            if nucleus_image.shape[-1] == 4:  # RGBA
                nucleus_image = nucleus_image[:, :, :3]
            
            # Create mask for this nucleus within the patch
            nucleus_mask = self._create_mask_from_polygon(
                nucleus_polygon,
                patch_center=(cx, cy),
                patch_size=self.patch_size
            )
            
        except Exception as e:
            print(f"    Warning: Could not read image for nucleus {nucleus_id}: {e}")
            # Return features with defaults
            return features
        
        # Extract features
        features.morphology = self._extract_morphology_features(nucleus_polygon)
        features.intensity = self._extract_intensity_features(nucleus_image, nucleus_mask)
        features.texture = self._extract_texture_features(nucleus_image, nucleus_mask)
        features.aging_markers = self._extract_aging_features(
            nucleus_image, nucleus_mask, nucleus_polygon
        )
        
        return features
    
    def _create_mask_from_polygon(
        self, 
        polygon: Polygon, 
        patch_center: Tuple[int, int],
        patch_size: int
    ) -> np.ndarray:
        """
        Create binary mask from polygon within image patch
        
        Args:
            polygon: Shapely Polygon in WSI coordinates
            patch_center: Center of patch in WSI coordinates (cx, cy)
            patch_size: Size of patch
        
        Returns:
            Binary mask (patch_size, patch_size)
        """
        # Translate polygon to patch coordinates
        cx, cy = patch_center
        half_patch = patch_size // 2
        
        # Shift polygon so patch center is at (half_patch, half_patch)
        polygon_shifted = translate(
            polygon,
            xoff=-cx + half_patch,
            yoff=-cy + half_patch
        )
        
        # Rasterize polygon
        mask = np.zeros((patch_size, patch_size), dtype=bool)
        
        # Get polygon coordinates
        if hasattr(polygon_shifted.exterior, 'coords'):
            coords = np.array(polygon_shifted.exterior.coords)
            coords = coords.astype(int)
            
            # Clip to image bounds
            coords[:, 0] = np.clip(coords[:, 0], 0, patch_size - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, patch_size - 1)
            
            # Fill polygon
            cv2.fillPoly(mask, [coords], True)
        
        return mask
    
    def _extract_morphology_features(self, nucleus_polygon: Polygon) -> Dict[str, float]:
        """Extract morphological features from nucleus polygon"""
        features = {}
        
        # Basic shape features
        features['area'] = float(nucleus_polygon.area)
        features['perimeter'] = float(nucleus_polygon.length)
        
        # Circularity
        features['circularity'] = (
            4 * np.pi * nucleus_polygon.area / (nucleus_polygon.length ** 2)
            if nucleus_polygon.length > 0 else 0
        )
        
        # Compactness (inverse circularity)
        features['compactness'] = (
            nucleus_polygon.length ** 2 / (4 * np.pi * nucleus_polygon.area)
            if nucleus_polygon.area > 0 else 0
        )
        
        # Convexity
        convex_hull = nucleus_polygon.convex_hull
        features['convexity'] = (
            nucleus_polygon.area / convex_hull.area 
            if convex_hull.area > 0 else 0
        )
        features['solidity'] = features['convexity']
        
        # Bounding box features
        minx, miny, maxx, maxy = nucleus_polygon.bounds
        bbox_width = maxx - minx
        bbox_height = maxy - miny
        
        features['bbox_width'] = float(bbox_width)
        features['bbox_height'] = float(bbox_height)
        features['aspect_ratio'] = (
            bbox_width / bbox_height if bbox_height > 0 else 0
        )
        
        # Extent (area / bounding box area)
        bbox_area = bbox_width * bbox_height
        features['extent'] = (
            nucleus_polygon.area / bbox_area if bbox_area > 0 else 0
        )
        
        # Irregularity
        features['irregularity'] = 1.0 - features['circularity']
        
        return features
    
    def _extract_intensity_features(
        self, 
        nucleus_image: np.ndarray, 
        nucleus_mask: np.ndarray
    ) -> Dict[str, float]:
        """Extract intensity features from nucleus"""
        features = {}
        
        # Convert to grayscale
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image
        
        # Get nucleus pixels
        nucleus_pixels = nucleus_gray[nucleus_mask]
        
        if len(nucleus_pixels) == 0:
            return {k: 0.0 for k in [
                'mean', 'std', 'min', 'max', 'range', 'median',
                'skewness', 'kurtosis', 'cv', 'entropy',
                'q10', 'q25', 'q75', 'q90'
            ]}
        
        # Basic statistics
        features['mean'] = float(np.mean(nucleus_pixels))
        features['std'] = float(np.std(nucleus_pixels))
        features['min'] = float(np.min(nucleus_pixels))
        features['max'] = float(np.max(nucleus_pixels))
        features['range'] = features['max'] - features['min']
        features['median'] = float(np.median(nucleus_pixels))
        
        # Higher order statistics
        features['skewness'] = float(self._skewness(nucleus_pixels))
        features['kurtosis'] = float(self._kurtosis(nucleus_pixels))
        
        # Coefficient of variation (chromatin heterogeneity)
        features['cv'] = (
            features['std'] / features['mean']
            if features['mean'] > 0 else 0
        )
        
        # Quantiles
        features['q10'] = float(np.percentile(nucleus_pixels, 10))
        features['q25'] = float(np.percentile(nucleus_pixels, 25))
        features['q75'] = float(np.percentile(nucleus_pixels, 75))
        features['q90'] = float(np.percentile(nucleus_pixels, 90))
        
        # Entropy
        hist, _ = np.histogram(nucleus_pixels, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        features['entropy'] = float(-np.sum(hist * np.log2(hist)))
        
        return features
    
    def _extract_texture_features(
        self,
        nucleus_image: np.ndarray,
        nucleus_mask: np.ndarray
    ) -> Dict[str, float]:
        """Extract texture features (chromatin patterns)"""
        features = {}
        
        # Convert to grayscale
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image.copy()
        
        # Mask the region
        nucleus_gray_masked = nucleus_gray.copy()
        nucleus_gray_masked[~nucleus_mask] = 0
        
        # Skip if too small
        if nucleus_mask.sum() < 10:
            return self._get_default_texture_features()
        
        try:
            # Normalize
            if nucleus_gray_masked.max() > nucleus_gray_masked.min():
                nucleus_gray_norm = (
                    (nucleus_gray_masked - nucleus_gray_masked.min()) / 
                    (nucleus_gray_masked.max() - nucleus_gray_masked.min()) * 255
                ).astype(np.uint8)
            else:
                return self._get_default_texture_features()
            
            # GLCM features
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(
                nucleus_gray_norm,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Extract properties
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                values = graycoprops(glcm, prop)
                features[f'glcm_{prop}_mean'] = float(np.mean(values))
                features[f'glcm_{prop}_std'] = float(np.std(values))
            
            # Haralick features using mahotas
            haralick = mahotas.features.haralick(nucleus_gray_norm, ignore_zeros=True)
            haralick_mean = haralick.mean(axis=0)
            
            haralick_names = [
                'angular_second_moment', 'contrast', 'correlation',
                'sum_of_squares', 'idm', 'sum_average', 'sum_variance',
                'sum_entropy', 'entropy', 'difference_variance',
                'difference_entropy', 'info_correlation_1', 'info_correlation_2'
            ]
            
            for name, value in zip(haralick_names, haralick_mean):
                features[f'haralick_{name}'] = float(value)
            
        except Exception as e:
            print(f"      Warning: Texture extraction failed: {e}")
            return self._get_default_texture_features()
        
        # Local Binary Patterns
        try:
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(nucleus_gray, n_points, radius, method='uniform')
            lbp_masked = lbp[nucleus_mask]
            
            if len(lbp_masked) > 0:
                features['lbp_mean'] = float(np.mean(lbp_masked))
                features['lbp_std'] = float(np.std(lbp_masked))
        except:
            features['lbp_mean'] = 0.0
            features['lbp_std'] = 0.0
        
        # Chromatin foci (local maxima)
        try:
            smoothed = ndimage.gaussian_filter(nucleus_gray.astype(float), sigma=1)
            local_max = (smoothed == ndimage.maximum_filter(smoothed, size=5))
            local_max = local_max & nucleus_mask
            features['chromatin_foci_count'] = int(np.sum(local_max))
            features['chromatin_foci_density'] = (
                features['chromatin_foci_count'] / nucleus_mask.sum()
                if nucleus_mask.sum() > 0 else 0
            )
        except:
            features['chromatin_foci_count'] = 0
            features['chromatin_foci_density'] = 0.0
        
        return features
    
    def _extract_aging_features(
        self,
        nucleus_image: np.ndarray,
        nucleus_mask: np.ndarray,
        nucleus_polygon: Polygon
    ) -> Dict[str, float]:
        """Extract aging-specific features (nucleus-only)"""
        features = {}
        
        # Convert to grayscale
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image
        
        nucleus_pixels = nucleus_gray[nucleus_mask]
        
        # 1. Nuclear irregularity (genomic instability)
        convex_hull = nucleus_polygon.convex_hull
        features['irregularity_score'] = 1.0 - (
            nucleus_polygon.area / convex_hull.area 
            if convex_hull.area > 0 else 0
        )
        features['is_irregular'] = float(features['irregularity_score'] > 0.1)
        
        # 2. Chromatin condensation (apoptosis/senescence marker)
        if len(nucleus_pixels) > 0:
            features['chromatin_condensation'] = float(np.mean(nucleus_pixels) / 255.0)
            features['is_condensed'] = float(np.mean(nucleus_pixels) > 180)
        else:
            features['chromatin_condensation'] = 0.0
            features['is_condensed'] = 0.0
        
        # 3. Nuclear fragmentation
        nucleus_labels = measure.label(nucleus_mask)
        n_fragments = len(np.unique(nucleus_labels)) - 1
        features['n_fragments'] = float(n_fragments)
        features['is_fragmented'] = float(n_fragments > 1)
        
        # 4. Nuclear size (pyknosis = small dense nucleus)
        features['is_pyknotic'] = float(
            (nucleus_polygon.area < 100) and 
            (len(nucleus_pixels) > 0 and np.mean(nucleus_pixels) > 180)
        )
        
        # 5. Composite apoptosis score (nucleus-based only)
        apoptosis_indicators = [
            features['is_fragmented'],
            features['is_condensed'],
            features['is_pyknotic']
        ]
        features['apoptosis_score'] = float(np.mean(apoptosis_indicators))
        
        return features
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add shape-invariant spatial context features
        
        Uses k-NN to compute local neighborhood statistics
        """
        if len(df) == 0:
            return df
            
        # Extract centroids
        centroids = df[['centroid_x', 'centroid_y']].values
        
        # Build k-NN graph
        k_neighbors = min(20, len(centroids) - 1)
        if k_neighbors < 1:
            # Too few nuclei, return without spatial features
            return df
        
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        # Compute spatial features for each nucleus
        spatial_features = []
        
        for i in range(len(centroids)):
            neighbor_distances = distances[i, 1:]  # Exclude self
            neighbor_indices = indices[i, 1:]
            neighbor_coords = centroids[neighbor_indices]
            
            # Relative coordinates
            relative_coords = neighbor_coords - centroids[i]
            
            # Spatial statistics (shape-invariant)
            features = {
                'local_density': k_neighbors / (np.pi * (neighbor_distances[-1] ** 2 + 1e-6)),
                'mean_neighbor_distance': np.mean(neighbor_distances),
                'std_neighbor_distance': np.std(neighbor_distances),
                'nearest_neighbor_dist': neighbor_distances[0],
                'nn_regularity': np.std(neighbor_distances) / (np.mean(neighbor_distances) + 1e-6),
                'neighbor_spread': np.std(np.linalg.norm(relative_coords, axis=1)),
                'neighbor_anisotropy': np.std(relative_coords[:, 0]) / (np.std(relative_coords[:, 1]) + 1e-6),
                'angular_std': np.std(np.arctan2(relative_coords[:, 1], relative_coords[:, 0])),
            }
            
            spatial_features.append(features)
        
        # Add to dataframe
        spatial_df = pd.DataFrame(spatial_features)
        spatial_df.columns = [f'spatial_{col}' for col in spatial_df.columns]
        
        df = pd.concat([df.reset_index(drop=True), spatial_df], axis=1)
        
        return df
    
    def _get_default_texture_features(self) -> Dict[str, float]:
        """Return default texture features when extraction fails"""
        features = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            features[f'glcm_{prop}_mean'] = 0.0
            features[f'glcm_{prop}_std'] = 0.0
        
        haralick_names = [
            'angular_second_moment', 'contrast', 'correlation',
            'sum_of_squares', 'idm', 'sum_average', 'sum_variance',
            'sum_entropy', 'entropy', 'difference_variance',
            'difference_entropy', 'info_correlation_1', 'info_correlation_2'
        ]
        for name in haralick_names:
            features[f'haralick_{name}'] = 0.0
        
        features['lbp_mean'] = 0.0
        features['lbp_std'] = 0.0
        features['chromatin_foci_count'] = 0
        features['chromatin_foci_density'] = 0.0
        
        return features
    
    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class GlobalGraphBuilder:
    """
    Build global graph from nucleus features
    Handles disconnected tissue components
    Uses shape-invariant features
    """
    
    def __init__(
        self,
        k_neighbors: int = 10,
        max_edge_distance: float = 500.0,
        use_laplacian_pe: bool = False,
        n_pe_dims: int = 16
    ):
        """
        Args:
            k_neighbors: Number of neighbors per node
            max_edge_distance: Maximum edge length (prevents cross-tissue edges)
            use_laplacian_pe: Whether to add graph Laplacian positional encodings
            n_pe_dims: Number of positional encoding dimensions
        """
        self.k_neighbors = k_neighbors
        self.max_edge_distance = max_edge_distance
        self.use_laplacian_pe = use_laplacian_pe
        self.n_pe_dims = n_pe_dims
    
    def build_graph(
        self,
        features_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Build global graph from nucleus features
        
        Args:
            features_df: DataFrame with nucleus features (must have centroid_x, centroid_y)
            output_path: Optional path to save graph
        
        Returns:
            Dictionary with node_features, edge_index, edge_attr
        """
        print("\nBuilding global graph...")
        print("=" * 60)
        
        if len(features_df) == 0:
            print("Warning: Empty features dataframe!")
            return {
                'node_features': np.array([]),
                'edge_index': np.array([]),
                'edge_attr': np.array([]),
                'num_nodes': 0,
                'feature_names': [],
                'centroids': np.array([])
            }
        
        # Extract centroids
        centroids = features_df[['centroid_x', 'centroid_y']].values
        N = len(centroids)
        print(f"Number of nodes: {N}")
        
        # Build k-NN graph
        print(f"Building k-NN graph (k={self.k_neighbors})...")
        edge_index, edge_features = self._build_knn_edges(centroids)
        print(f"Number of edges: {len(edge_index)}")
        
        # Prepare node features (REMOVE centroid_x, centroid_y)
        node_feature_cols = [
            col for col in features_df.columns 
            if col not in ['nucleus_id', 'centroid_x', 'centroid_y']
        ]
        node_features = features_df[node_feature_cols].values
        print(f"Node feature dimensions: {node_features.shape[1]}")
        
        # Optional: Add graph positional encodings
        if self.use_laplacian_pe and len(edge_index) > 0:
            print(f"Computing Laplacian PE ({self.n_pe_dims} dims)...")
            pe = self._compute_laplacian_pe(edge_index, N)
            node_features = np.hstack([node_features, pe])
            print(f"Node features with PE: {node_features.shape[1]} dims")
        
        # Create graph dictionary
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_features,
            'num_nodes': N,
            'feature_names': node_feature_cols,
            'centroids': centroids  # Keep for visualization
        }
        
        # Save if requested
        if output_path is not None:
            self._save_graph(graph_data, output_path)
        
        print("\n✓ Graph construction complete!")
        return graph_data
    
    def _build_knn_edges(
        self, 
        centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN edges with distance threshold
        
        Returns:
            edge_index: (E, 2) array of edge indices
            edge_features: (E, 4) array of [distance, angle, dx_norm, dy_norm]
        """
        N = len(centroids)
        k = min(self.k_neighbors, N - 1)
        
        if k < 1:
            return np.array([]), np.array([])
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        edge_list = []
        edge_feat_list = []
        
        for i in range(N):
            for j, neighbor_idx in enumerate(indices[i, 1:]):  # Skip self
                dist = distances[i, j + 1]
                
                # Skip edges that are too long
                if dist > self.max_edge_distance:
                    continue
                
                # Compute edge features
                vec = centroids[neighbor_idx] - centroids[i]
                angle = np.arctan2(vec[1], vec[0])
                
                edge_list.append([i, neighbor_idx])
                edge_feat_list.append([
                    dist,
                    angle,
                    vec[0] / (dist + 1e-6),  # Normalized direction
                    vec[1] / (dist + 1e-6)
                ])
        
        if len(edge_list) == 0:
            return np.array([]), np.array([])
        
        edge_index = np.array(edge_list, dtype=np.int64)
        edge_features = np.array(edge_feat_list, dtype=np.float32)
        
        return edge_index, edge_features
    
    def _compute_laplacian_pe(
        self,
        edge_index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Compute graph Laplacian positional encodings
        Handles disconnected components
        """
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components, laplacian
        
        # Build adjacency matrix
        row, col = edge_index[:, 0], edge_index[:, 1]
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        adj = adj + adj.T  # Symmetric
        
        # Find connected components
        n_components, labels = connected_components(adj, directed=False)
        print(f"  Found {n_components} connected component(s)")
        
        # Initialize PE matrix
        pe_matrix = np.zeros((num_nodes, self.n_pe_dims))
        
        # Compute PE for each component
        for comp_id in range(n_components):
            mask = labels == comp_id
            comp_indices = np.where(mask)[0]
            
            if len(comp_indices) < 2:
                continue
            
            # Extract subgraph
            sub_adj = adj[mask][:, mask]
            
            # Compute normalized Laplacian
            lap = laplacian(sub_adj, normed=True)
            lap_dense = lap.toarray()
            
            try:
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(lap_dense)
                
                # Use smallest non-zero eigenvectors
                n_eig = min(self.n_pe_dims, len(eigenvalues) - 1)
                comp_pe = eigenvectors[:, 1:n_eig + 1]
                
                # Pad if needed
                if comp_pe.shape[1] < self.n_pe_dims:
                    padding = np.zeros((comp_pe.shape[0], self.n_pe_dims - comp_pe.shape[1]))
                    comp_pe = np.hstack([comp_pe, padding])
                
                pe_matrix[comp_indices] = comp_pe
                
            except np.linalg.LinAlgError:
                print(f"  Warning: Failed to compute PE for component {comp_id}")
                continue
        
        return pe_matrix
    
    def _save_graph(self, graph_data: Dict, output_path: str):
        """Save graph data to disk"""
        import pickle
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"\n✓ Saved graph to {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Example usage of Phase 1 pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1: Nucleus-only graph construction')
    parser.add_argument('wsi_path', type=str, help='Path to WSI file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size')
    parser.add_argument('--mpp', type=float, default=0.5, help='Microns per pixel')
    parser.add_argument('--batch_size', type=int, default=32, help='InstanSeg batch size')
    parser.add_argument('--k_neighbors', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--max_distance', type=float, default=500, help='Max edge distance')
    parser.add_argument('--use_pe', action='store_true', help='Use Laplacian PE')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract nucleus features
    print("\n" + "="*60)
    print("PHASE 1: NUCLEUS-ONLY GRAPH CONSTRUCTION")
    print("="*60)
    
    extractor = LazySlideNucleusExtractor(
        patch_size=128,
        min_nucleus_area=20,
        max_nucleus_area=10000
    )
    
    features_df = extractor.process_wsi(
        wsi_path=args.wsi_path,
        output_path=output_dir / 'nucleus_features.csv',
        tile_size=args.tile_size,
        mpp=args.mpp,
        batch_size=args.batch_size
    )
    
    # Step 2: Build global graph
    graph_builder = GlobalGraphBuilder(
        k_neighbors=args.k_neighbors,
        max_edge_distance=args.max_distance,
        use_laplacian_pe=args.use_pe,
        n_pe_dims=16
    )
    
    graph_data = graph_builder.build_graph(
        features_df=features_df,
        output_path=output_dir / 'nucleus_graph.pkl'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Nuclei processed: {graph_data['num_nodes']}")
    print(f"Edges created: {len(graph_data['edge_index'])}")
    print(f"Node feature dimensions: {graph_data['node_features'].shape[1] if len(graph_data['node_features']) > 0 else 0}")
    print(f"Features saved to: {output_dir / 'nucleus_features.csv'}")
    print(f"Graph saved to: {output_dir / 'nucleus_graph.pkl'}")
    
    if graph_data['num_nodes'] > 0:
        print("\nFeature categories:")
        feature_names = graph_data['feature_names']
        for prefix in ['morph_', 'intensity_', 'texture_', 'aging_', 'spatial_']:
            count = sum(1 for name in feature_names if name.startswith(prefix))
            print(f"  {prefix[:-1]}: {count} features")
    
    return features_df, graph_data


if __name__ == '__main__':
    main()