import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')
import lazyslide as zs
# Image processing
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
from scipy.spatial import distance
import mahotas
from wsidata import open_wsi
import argparse


@dataclass
class CellFeatures:
    """Container for all cell-level features"""
    cell_id: int
    centroid_x: float
    centroid_y: float
    
    # Morphological features
    morphology: Dict[str, float] = field(default_factory=dict)
    
    # Intensity features
    intensity: Dict[str, float] = field(default_factory=dict)
    
    # Texture features
    texture: Dict[str, float] = field(default_factory=dict)
    
    # Aging-specific features
    aging_markers: Dict[str, float] = field(default_factory=dict)
    
    # Cell type prediction
    # Need an actual cell type prediction model for now
    # cell_type: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to flat dictionary for DataFrame"""
        result = {
            'cell_id': self.cell_id,
            'centroid_x': self.centroid_x,
            'centroid_y': self.centroid_y,
        }
        result.update({f'morph_{k}': v for k, v in self.morphology.items()})
        result.update({f'intensity_{k}': v for k, v in self.intensity.items()})
        result.update({f'texture_{k}': v for k, v in self.texture.items()})
        result.update({f'aging_{k}': v for k, v in self.aging_markers.items()})
        # result.update({f'celltype_{k}': v for k, v in self.cell_type.items()})
        
        return result


class WSIFeatureExtractor:
    """
    Extract comprehensive features from segmented cells in whole slide images
    """
    
    def __init__(
        self,
        patch_size: int = 224,
    ):
        """
        Initialize feature extractor
        
        Args:
            patch_size: Size of patches to extract around cells
        """
        self.patch_size = patch_size
    
    
    def process_wsi(
        self,
        wsi_path: Union[str, Path],
        nucleus_mask_path: Union[str, Path],
        cell_mask_path: Optional[Union[str, Path]] = None,
        min_nucleus_area: int = 20,
        max_nucleus_area: int = 10000,
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Process entire WSI and extract features for all cells
        
        Args:
            wsi_path: Path to whole slide image
            nucleus_mask_path: Path to nucleus segmentation mask
            cell_mask_path: Optional path to cell segmentation mask
            level: Pyramid level to process (0 = highest resolution)
            min_nucleus_area: Minimum nucleus area (pixels)
            max_nucleus_area: Maximum nucleus area (pixels)
            output_path: Optional path to save results
        
        Returns:
            DataFrame with all cell features
        """
        print(f"Processing WSI: {wsi_path}")
        
        # Load images
        wsi_image = open_wsi(wsi_path)
        nucleus_mask = self._load_mask(nucleus_mask_path)
        cell_mask = self._load_mask(cell_mask_path)
    
        # Ensure masks match image size
        assert wsi_image.shape[:2] == nucleus_mask.shape[:2], \
            "WSI and mask dimensions don't match"
        
        # Extract features for all cells
        all_features = []
        
        # Get nucleus regions
        nucleus_props = measure.regionprops(nucleus_mask)
        print(f"Found {len(nucleus_props)} nuclei")
        
        # Filter by size
        nucleus_props = [
            prop for prop in nucleus_props 
            if min_nucleus_area <= prop.area <= max_nucleus_area
        ]
        print(f"After filtering: {len(nucleus_props)} nuclei")
        
        # Process each cell
        for idx, nucleus_prop in enumerate(nucleus_props):
            if idx % 100 == 0:
                print(f"Processing cell {idx}/{len(nucleus_props)}")
            
            try:
                # Get corresponding cell region
                cell_prop = self._get_cell_for_nucleus(nucleus_prop, cell_mask)
                
                if cell_prop is None:
                    continue
                
                # Extract features
                features = self.extract_cell_features(
                    wsi_image=wsi_image,
                    nucleus_prop=nucleus_prop,
                    cell_prop=cell_prop,
                    nucleus_mask=nucleus_mask,
                    cell_mask=cell_mask,
                    cell_id=idx
                )
                
                all_features.append(features)
                
            except Exception as e:
                print(f"Error processing cell {idx}: {str(e)}")
                continue
        
        # Convert to DataFrame
        print(f"Successfully extracted features from {len(all_features)} cells")
        df = pd.DataFrame([f.to_dict() for f in all_features])
        
        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved features to {output_path}")
        
        return df
    
    def extract_cell_features(
        self,
        wsi_image: np.ndarray,
        nucleus_prop: measure._regionprops.RegionProperties,
        cell_prop: measure._regionprops.RegionProperties,
        nucleus_mask: np.ndarray,
        cell_mask: np.ndarray,
        cell_id: int
    ) -> CellFeatures:
        """
        Extract all features for a single cell
        
        Args:
            wsi_image: Full WSI image (H, W, 3)
            nucleus_prop: Regionprops for nucleus
            cell_prop: Regionprops for cell
            nucleus_mask: Full nucleus segmentation mask
            cell_mask: Full cell segmentation mask
            cell_id: Unique cell identifier
        
        Returns:
            CellFeatures object with all extracted features
        """
        features = CellFeatures(
            cell_id=cell_id,
            centroid_x=cell_prop.centroid[1],
            centroid_y=cell_prop.centroid[0]
        )
        
        # Extract bounding box regions
        nucleus_bbox = nucleus_prop.bbox
        cell_bbox = cell_prop.bbox
        
        # Get image patches
        nucleus_image = wsi_image[
            nucleus_bbox[0]:nucleus_bbox[2],
            nucleus_bbox[1]:nucleus_bbox[3]
        ]
        cell_image = wsi_image[
            cell_bbox[0]:cell_bbox[2],
            cell_bbox[1]:cell_bbox[3]
        ]
        
        # Get mask patches
        nucleus_mask_patch = (
            nucleus_mask[nucleus_bbox[0]:nucleus_bbox[2], nucleus_bbox[1]:nucleus_bbox[3]]
            == nucleus_prop.label
        )
        cell_mask_patch = (
            cell_mask[cell_bbox[0]:cell_bbox[2], cell_bbox[1]:cell_bbox[3]]
            == cell_prop.label
        )
        
        # 1. Morphological features
        features.morphology = self._extract_morphology_features(
            nucleus_prop, cell_prop
        )
        
        # 2. Intensity features
        features.intensity = self._extract_intensity_features(
            nucleus_image, cell_image
        )
        
        # 3. Texture features
        features.texture = self._extract_texture_features(
            nucleus_image, nucleus_mask_patch
        )
        
        # 4. Aging-specific features
        features.aging_markers = self._extract_aging_features(
            nucleus_image, cell_image, nucleus_prop, cell_prop,
            nucleus_mask_patch, cell_mask_patch
        )
        
        # 5. Cell type classification features
        # features.cell_type = self._classify_cell_type(
        #     nucleus_prop, cell_prop, features.morphology, features.intensity
        # )
        
        return features
    
    def _extract_morphology_features(
        self,
        nucleus_prop: measure._regionprops.RegionProperties,
        cell_prop: measure._regionprops.RegionProperties
    ) -> Dict[str, float]:
        """Extract morphological features from nucleus and cell"""
        features = {}
        
        # === Nuclear Morphology ===
        features['nucleus_area'] = nucleus_prop.area
        features['nucleus_perimeter'] = nucleus_prop.perimeter
        features['nucleus_circularity'] = (
            4 * np.pi * nucleus_prop.area / (nucleus_prop.perimeter ** 2)
            if nucleus_prop.perimeter > 0 else 0
        )
        features['nucleus_eccentricity'] = nucleus_prop.eccentricity
        features['nucleus_solidity'] = nucleus_prop.solidity
        features['nucleus_extent'] = nucleus_prop.extent
        features['nucleus_major_axis'] = nucleus_prop.major_axis_length
        features['nucleus_minor_axis'] = nucleus_prop.minor_axis_length
        features['nucleus_aspect_ratio'] = (
            nucleus_prop.major_axis_length / nucleus_prop.minor_axis_length
            if nucleus_prop.minor_axis_length > 0 else 0
        )
        
        # Compactness (inverse of circularity)
        features['nucleus_compactness'] = (
            nucleus_prop.perimeter ** 2 / (4 * np.pi * nucleus_prop.area)
            if nucleus_prop.area > 0 else 0
        )
        
        # Convexity
        convex_area = nucleus_prop.convex_area
        features['nucleus_convexity'] = (
            nucleus_prop.area / convex_area if convex_area > 0 else 0
        )
        
        # Boundary irregularity
        features['nucleus_irregularity'] = 1.0 - features['nucleus_circularity']
        
        # === Cell Morphology ===
        features['cell_area'] = cell_prop.area
        features['cell_perimeter'] = cell_prop.perimeter
        features['cell_circularity'] = (
            4 * np.pi * cell_prop.area / (cell_prop.perimeter ** 2)
            if cell_prop.perimeter > 0 else 0
        )
        features['cell_eccentricity'] = cell_prop.eccentricity
        features['cell_solidity'] = cell_prop.solidity
        features['cell_major_axis'] = cell_prop.major_axis_length
        features['cell_minor_axis'] = cell_prop.minor_axis_length
        features['cell_aspect_ratio'] = (
            cell_prop.major_axis_length / cell_prop.minor_axis_length
            if cell_prop.minor_axis_length > 0 else 0
        )
        
        # === Combined Features ===
        # Cytoplasmic area (cell - nucleus)
        features['cytoplasm_area'] = cell_prop.area - nucleus_prop.area
        
        # Nuclear-to-Cytoplasmic ratio (KEY SENESCENCE MARKER)
        features['nc_ratio'] = (
            nucleus_prop.area / features['cytoplasm_area']
            if features['cytoplasm_area'] > 0 else 0
        )
        
        # Cell-to-nucleus area ratio
        features['cell_nucleus_ratio'] = (
            cell_prop.area / nucleus_prop.area
            if nucleus_prop.area > 0 else 0
        )
        
        # Nuclear position within cell (eccentricity)
        nucleus_centroid = np.array(nucleus_prop.centroid)
        cell_centroid = np.array(cell_prop.centroid)
        features['nucleus_eccentricity_in_cell'] = np.linalg.norm(
            nucleus_centroid - cell_centroid
        )
        
        return features
    
    def _extract_intensity_features(
        self,
        nucleus_image: np.ndarray,
        cell_image: np.ndarray,
        nucleus_mask: np.ndarray,
        cell_mask: np.ndarray
    ) -> Dict[str, float]:
        """Extract intensity-based features"""
        features = {}
        
        # Convert to grayscale if needed
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image
            
        if len(cell_image.shape) == 3:
            cell_gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
        else:
            cell_gray = cell_image
        
        # === Nuclear Intensity ===
        nucleus_pixels = nucleus_gray[nucleus_mask]
        
        if len(nucleus_pixels) > 0:
            features['nucleus_intensity_mean'] = float(np.mean(nucleus_pixels))
            features['nucleus_intensity_std'] = float(np.std(nucleus_pixels))
            features['nucleus_intensity_min'] = float(np.min(nucleus_pixels))
            features['nucleus_intensity_max'] = float(np.max(nucleus_pixels))
            features['nucleus_intensity_range'] = features['nucleus_intensity_max'] - features['nucleus_intensity_min']
            features['nucleus_intensity_median'] = float(np.median(nucleus_pixels))
            
            # Higher order statistics
            features['nucleus_intensity_skewness'] = float(
                self._skewness(nucleus_pixels)
            )
            features['nucleus_intensity_kurtosis'] = float(
                self._kurtosis(nucleus_pixels)
            )
            
            # Coefficient of variation (chromatin heterogeneity)
            features['nucleus_intensity_cv'] = (
                features['nucleus_intensity_std'] / features['nucleus_intensity_mean']
                if features['nucleus_intensity_mean'] > 0 else 0
            )
            
            # Quantiles
            features['nucleus_intensity_q10'] = float(np.percentile(nucleus_pixels, 10))
            features['nucleus_intensity_q25'] = float(np.percentile(nucleus_pixels, 25))
            features['nucleus_intensity_q75'] = float(np.percentile(nucleus_pixels, 75))
            features['nucleus_intensity_q90'] = float(np.percentile(nucleus_pixels, 90))
            
            # Entropy (randomness)
            hist, _ = np.histogram(nucleus_pixels, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            features['nucleus_intensity_entropy'] = float(-np.sum(hist * np.log2(hist)))
        
        # === Cytoplasmic Intensity ===
        # Get cytoplasm mask (cell - nucleus)
        cytoplasm_mask = cell_mask.copy()
        
        # Need to align nucleus mask with cell mask
        cell_h, cell_w = cell_mask.shape
        nucleus_h, nucleus_w = nucleus_mask.shape
        
        # Create aligned nucleus mask
        aligned_nucleus_mask = np.zeros_like(cell_mask, dtype=bool)
        
        # This is approximate - assumes similar positioning
        # In practice, you'd need to track actual coordinates
        if nucleus_h <= cell_h and nucleus_w <= cell_w:
            start_h = (cell_h - nucleus_h) // 2
            start_w = (cell_w - nucleus_w) // 2
            aligned_nucleus_mask[
                start_h:start_h+nucleus_h,
                start_w:start_w+nucleus_w
            ] = nucleus_mask
            
            cytoplasm_mask = cell_mask & ~aligned_nucleus_mask
        
        cytoplasm_pixels = cell_gray[cytoplasm_mask]
        
        if len(cytoplasm_pixels) > 0:
            features['cytoplasm_intensity_mean'] = float(np.mean(cytoplasm_pixels))
            features['cytoplasm_intensity_std'] = float(np.std(cytoplasm_pixels))
            features['cytoplasm_intensity_cv'] = (
                features['cytoplasm_intensity_std'] / features['cytoplasm_intensity_mean']
                if features['cytoplasm_intensity_mean'] > 0 else 0
            )
            
            # Cytoplasmic eosinophilia (pink staining in H&E)
            # Higher intensity = more eosinophilic (common in senescence)
            if len(cell_image.shape) == 3:
                # Red channel often indicates eosinophilia in H&E
                cytoplasm_red = cell_image[:, :, 0][cytoplasm_mask]
                features['cytoplasm_eosinophilia'] = float(np.mean(cytoplasm_red))
        
        # === Nuclear-Cytoplasmic Contrast ===
        if len(nucleus_pixels) > 0 and len(cytoplasm_pixels) > 0:
            features['nucleus_cytoplasm_contrast'] = (
                features['nucleus_intensity_mean'] - features['cytoplasm_intensity_mean']
            )
            features['nucleus_cytoplasm_ratio'] = (
                features['nucleus_intensity_mean'] / features['cytoplasm_intensity_mean']
                if features['cytoplasm_intensity_mean'] > 0 else 0
            )
        
        # === Color Features (if RGB) ===
        if len(cell_image.shape) == 3:
            cell_pixels_rgb = cell_image[cell_mask]
            
            if len(cell_pixels_rgb) > 0:
                features['cell_red_mean'] = float(np.mean(cell_pixels_rgb[:, 0]))
                features['cell_green_mean'] = float(np.mean(cell_pixels_rgb[:, 1]))
                features['cell_blue_mean'] = float(np.mean(cell_pixels_rgb[:, 2]))
                
                # Color ratios
                features['cell_rg_ratio'] = (
                    features['cell_red_mean'] / features['cell_green_mean']
                    if features['cell_green_mean'] > 0 else 0
                )
                features['cell_rb_ratio'] = (
                    features['cell_red_mean'] / features['cell_blue_mean']
                    if features['cell_blue_mean'] > 0 else 0
                )
        
        return features
    
    def _extract_texture_features(
        self,
        nucleus_image: np.ndarray,
        nucleus_mask: np.ndarray
    ) -> Dict[str, float]:
        """Extract texture features from nucleus (chromatin patterns)"""
        features = {}
        
        # Convert to grayscale
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image.copy()
        
        # Mask the region
        nucleus_gray = nucleus_gray.copy()
        nucleus_gray[~nucleus_mask] = 0
        
        # === Haralick Features ===
        try:
            # Normalize to 0-255
            nucleus_gray_norm = ((nucleus_gray - nucleus_gray.min()) / 
                                 (nucleus_gray.max() - nucleus_gray.min() + 1e-8) * 255).astype(np.uint8)
            
            # Compute GLCM (Gray Level Co-occurrence Matrix)
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
            
            # Extract Haralick properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 
                         'energy', 'correlation', 'ASM']
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                features[f'haralick_{prop}_mean'] = float(np.mean(values))
                features[f'haralick_{prop}_std'] = float(np.std(values))
            
            # Additional Haralick features using mahotas
            haralick_mahotas = mahotas.features.haralick(nucleus_gray_norm, ignore_zeros=True)
            haralick_mean = haralick_mahotas.mean(axis=0)
            
            haralick_names = [
                'angular_second_moment', 'contrast', 'correlation',
                'sum_of_squares_variance', 'inverse_difference_moment',
                'sum_average', 'sum_variance', 'sum_entropy',
                'entropy', 'difference_variance', 'difference_entropy',
                'info_measure_correlation_1', 'info_measure_correlation_2'
            ]
            
            for name, value in zip(haralick_names, haralick_mean):
                features[f'haralick_mahotas_{name}'] = float(value)
                
        except Exception as e:
            print(f"Warning: Haralick feature extraction failed: {str(e)}")
            # Fill with zeros
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                features[f'haralick_{prop}_mean'] = 0.0
                features[f'haralick_{prop}_std'] = 0.0
        
        # === Local Binary Patterns (LBP) ===
        try:
            # LBP parameters
            radius = 1
            n_points = 8 * radius
            
            lbp = local_binary_pattern(nucleus_gray, n_points, radius, method='uniform')
            lbp_masked = lbp[nucleus_mask]
            
            if len(lbp_masked) > 0:
                # Histogram of LBP patterns
                n_bins = int(lbp_masked.max() + 1)
                hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins))
                hist = hist.astype(float) / hist.sum()
                
                # Statistics of LBP histogram
                features['lbp_mean'] = float(np.mean(lbp_masked))
                features['lbp_std'] = float(np.std(lbp_masked))
                features['lbp_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
                features['lbp_uniformity'] = float(np.sum(hist ** 2))
                
        except Exception as e:
            print(f"Warning: LBP extraction failed: {str(e)}")
            features['lbp_mean'] = 0.0
            features['lbp_std'] = 0.0
            features['lbp_entropy'] = 0.0
            features['lbp_uniformity'] = 0.0
        
        # === Fractal Dimension ===
        try:
            features['fractal_dimension'] = self._fractal_dimension(nucleus_gray, nucleus_mask)
        except:
            features['fractal_dimension'] = 0.0
        
        # === Chromatin Granularity ===
        try:
            # Find local maxima (chromatin clumps)
            from scipy import ndimage as ndi
            
            # Smooth image slightly
            smoothed = ndi.gaussian_filter(nucleus_gray.astype(float), sigma=1)
            
            # Find local maxima
            local_max = (smoothed == ndi.maximum_filter(smoothed, size=5))
            local_max = local_max & nucleus_mask
            
            features['chromatin_foci_count'] = int(np.sum(local_max))
            
            # Chromatin foci density
            features['chromatin_foci_density'] = (
                features['chromatin_foci_count'] / nucleus_mask.sum()
                if nucleus_mask.sum() > 0 else 0
            )
            
        except Exception as e:
            features['chromatin_foci_count'] = 0
            features['chromatin_foci_density'] = 0.0
        
        return features
    
    def _extract_aging_features(
        self,
        nucleus_image: np.ndarray,
        cell_image: np.ndarray,
        nucleus_prop: measure._regionprops.RegionProperties,
        cell_prop: measure._regionprops.RegionProperties,
        nucleus_mask: np.ndarray,
        cell_mask: np.ndarray
    ) -> Dict[str, float]:
        """Extract aging-specific features"""
        features = {}
        
        # === Senescence Markers ===
        
        # 1. Cell enlargement (senescent cells are 2-3x larger)
        features['is_enlarged'] = float(cell_prop.area > 1000)  # Threshold depends on magnification
        features['enlargement_score'] = float(cell_prop.area / 500)  # Relative to typical size
        
        # 2. Low N/C ratio (KEY senescence marker)
        cytoplasm_area = cell_prop.area - nucleus_prop.area
        nc_ratio = nucleus_prop.area / cytoplasm_area if cytoplasm_area > 0 else 0
        features['is_low_nc_ratio'] = float(nc_ratio < 0.3)  # Senescent cells have low N/C
        
        # 3. Lipofuscin detection (brown/yellow granules - CRITICAL MARKER)
        lipofuscin_score = self._detect_lipofuscin(cell_image, cell_mask, nucleus_mask)
        features['lipofuscin_score'] = lipofuscin_score
        features['has_lipofuscin'] = float(lipofuscin_score > 0.1)
        
        # 4. Nuclear irregularity (genomic instability)
        features['nuclear_irregularity_score'] = 1.0 - nucleus_prop.solidity
        features['is_irregular_nucleus'] = float(nucleus_prop.solidity < 0.9)
        
        # 5. Chromatin condensation (apoptosis/senescence marker)
        if len(nucleus_image.shape) == 3:
            nucleus_gray = cv2.cvtColor(nucleus_image, cv2.COLOR_RGB2GRAY)
        else:
            nucleus_gray = nucleus_image
        
        nucleus_pixels = nucleus_gray[nucleus_mask]
        if len(nucleus_pixels) > 0:
            # High intensity = condensed chromatin
            features['chromatin_condensation'] = float(np.mean(nucleus_pixels) / 255.0)
            features['is_condensed_chromatin'] = float(np.mean(nucleus_pixels) > 180)
        
        # === Apoptotic Features ===
        
        # 6. Nuclear fragmentation
        # Label connected components in nucleus
        nucleus_labels = measure.label(nucleus_mask)
        n_fragments = len(np.unique(nucleus_labels)) - 1  # Subtract background
        features['nuclear_fragments'] = float(n_fragments)
        features['is_fragmented'] = float(n_fragments > 1)
        
        # 7. Cell shrinkage (apoptotic cells shrink)
        features['is_shrunken'] = float(cell_prop.area < 200)  # Threshold depends on magnification
        
        # 8. Pyknosis (small, dense nucleus)
        features['is_pyknotic'] = float(
            (nucleus_prop.area < 100) and (np.mean(nucleus_pixels) > 180)
        )
        
        # === Composite Aging Score ===
        # Combine multiple markers for overall aging assessment
        senescence_indicators = [
            features['is_enlarged'],
            features['is_low_nc_ratio'],
            features['has_lipofuscin'],
            features['is_irregular_nucleus']
        ]
        features['senescence_score'] = float(np.mean(senescence_indicators))
        
        apoptosis_indicators = [
            features['is_fragmented'],
            features['is_condensed_chromatin'],
            features['is_shrunken']
        ]
        features['apoptosis_score'] = float(np.mean(apoptosis_indicators))
        
        # Overall aging score
        features['aging_score'] = (
            0.6 * features['senescence_score'] + 
            0.4 * features['apoptosis_score']
        )
        
        return features
    
    def _detect_lipofuscin(
        self,
        cell_image: np.ndarray,
        cell_mask: np.ndarray,
    ) -> float:
        """
        Detect lipofuscin granules (brown/yellow pigment - KEY aging marker)
        
        Lipofuscin appears as brown/yellow granules in H&E staining
        """
        if len(cell_image.shape) != 3:
            return 0.0
        
        # Get cytoplasm mask (cell - nucleus)
        # Align masks (simplified version)
        cytoplasm_mask = cell_mask.copy()
        
        # Convert to different color spaces for better detection
        # RGB
        cell_rgb = cell_image.copy()
        
        # HSV (Hue-Saturation-Value)
        cell_hsv = cv2.cvtColor(cell_image, cv2.COLOR_RGB2HSV)
                
        # Lipofuscin detection criteria:
        # - Brown/yellow color (high red, moderate green, low blue)
        # - Granular appearance (local intensity peaks)
        
        # Color-based detection
        R = cell_rgb[:, :, 0].astype(float)
        G = cell_rgb[:, :, 1].astype(float)
        B = cell_rgb[:, :, 2].astype(float)
        
        # Brown/yellow: R > G > B
        brown_mask = (R > G) & (G > B) & (R > 100) & cytoplasm_mask
        
        # HSV-based (yellow-brown hue range: 20-40 degrees)
        H = cell_hsv[:, :, 0]
        S = cell_hsv[:, :, 1]
        V = cell_hsv[:, :, 2]
        
        # Yellow-brown hue range in OpenCV (0-180 scale)
        hue_mask = ((H >= 10) & (H <= 30) & (S > 50) & (V > 50)) & cytoplasm_mask
        
        # Combine masks
        lipofuscin_mask = brown_mask | hue_mask
        
        # Granule detection (size filtering)
        lipofuscin_labeled = measure.label(lipofuscin_mask)
        lipofuscin_props = measure.regionprops(lipofuscin_labeled)
        
        # Filter by size (granules are typically 1-10 pixels in diameter)
        valid_granules = [
            prop for prop in lipofuscin_props
            if 5 <= prop.area <= 100  # Adjust based on magnification
        ]
        
        # Lipofuscin score = (area of granules) / (cytoplasm area)
        if cytoplasm_mask.sum() > 0:
            total_granule_area = sum(prop.area for prop in valid_granules)
            lipofuscin_score = total_granule_area / cytoplasm_mask.sum()
        else:
            lipofuscin_score = 0.0
        
        return float(lipofuscin_score)
    
    def _classify_cell_type(
        self,
        nucleus_prop: measure._regionprops.RegionProperties,
        cell_prop: measure._regionprops.RegionProperties,
        morphology_features: Dict[str, float],
        intensity_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Classify cell type based on morphological features
        
        Returns probability scores for different cell types
        """
        pass
        # scores = {}
        
        # # Extract relevant features
        # cell_area = cell_prop.area
        # nucleus_area = nucleus_prop.area
        # nc_ratio = morphology_features.get('nc_ratio', 0)
        # nucleus_intensity = intensity_features.get('nucleus_intensity_mean', 0)
        # cell_circularity = morphology_features.get('cell_circularity', 0)
        
        # # === Oocyte Detection ===
        # # Very large cell, very large nucleus, circular
        # oocyte_score = 0.0
        # if cell_area > 2000:  # Very large
        #     oocyte_score += 0.3
        # if nucleus_area > 400:  # Very large nucleus
        #     oocyte_score += 0.3
        # if cell_circularity > 0.8:  # Circular
        #     oocyte_score += 0.2
        # if nc_ratio > 0.3:  # High N/C ratio
        #     oocyte_score += 0.2
        # scores['oocyte'] = min(oocyte_score, 1.0)
        
        # # === Granulosa Cell Detection ===
        # # Small, round, moderate N/C ratio
        # granulosa_score = 0.0
        # if 100 < cell_area < 500:  # Small-medium
        #     granulosa_score += 0.3
        # if 50 < nucleus_area < 200:  # Small-medium nucleus
        #     granulosa_score += 0.3
        # if 0.2 < nc_ratio < 0.5:  # Moderate N/C
        #     granulosa_score += 0.2
        # if cell_circularity > 0.7:  # Roundish
        #     granulosa_score += 0.2
        # scores['granulosa'] = min(granulosa_score, 1.0)
        
        # # === Lymphocyte Detection ===
        # # Very small, very high N/C ratio, dense nucleus
        # lymphocyte_score = 0.0
        # if cell_area < 150:  # Very small
        #     lymphocyte_score += 0.3
        # if nc_ratio > 0.7:  # Very high N/C (almost all nucleus)
        #     lymphocyte_score += 0.3
        # if nucleus_intensity > 150:  # Dense, dark nucleus
        #     lymphocyte_score += 0.2
        # if cell_circularity > 0.8:  # Round
        #     lymphocyte_score += 0.2
        # scores['lymphocyte'] = min(lymphocyte_score, 1.0)
        
        # # === Macrophage Detection ===
        # # Medium-large, irregular shape, moderate N/C
        # macrophage_score = 0.0
        # if 300 < cell_area < 1000:  # Medium-large
        #     macrophage_score += 0.3
        # if 0.15 < nc_ratio < 0.4:  # Moderate N/C
        #     macrophage_score += 0.2
        # if cell_circularity < 0.7:  # Irregular
        #     macrophage_score += 0.3
        # if morphology_features.get('nucleus_eccentricity', 0) > 0.5:  # Kidney-shaped nucleus
        #     macrophage_score += 0.2
        # scores['macrophage'] = min(macrophage_score, 1.0)
        
        # # === Stromal/Fibroblast Detection ===
        # # Elongated cells, elongated nuclei
        # stromal_score = 0.0
        # cell_aspect_ratio = morphology_features.get('cell_aspect_ratio', 1)
        # nucleus_aspect_ratio = morphology_features.get('nucleus_aspect_ratio', 1)
        
        # if cell_aspect_ratio > 2.0:  # Elongated cell
        #     stromal_score += 0.4
        # if nucleus_aspect_ratio > 1.5:  # Elongated nucleus
        #     stromal_score += 0.3
        # if nc_ratio < 0.3:  # Low N/C
        #     stromal_score += 0.3
        # scores['stromal'] = min(stromal_score, 1.0)
        
        # # Normalize scores
        # total_score = sum(scores.values())
        # if total_score > 0:
        #     scores = {k: v / total_score for k, v in scores.items()}
        
        # # Add "unknown" category
        # max_score = max(scores.values()) if scores else 0
        # scores['unknown'] = 1.0 - max_score
        
        # return scores
    
    # === Helper Methods ===
    
    def _load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Load segmentation mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        # Handle different formats
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel
        
        return mask.astype(np.int32)
    
    def _get_cell_for_nucleus(
        self,
        nucleus_prop: measure._regionprops.RegionProperties,
        cell_mask: np.ndarray,
    ) -> Optional[measure._regionprops.RegionProperties]:
        """Find corresponding cell for a nucleus"""
        # Get nucleus centroid
        cy, cx = nucleus_prop.centroid
        cy, cx = int(cy), int(cx)
        
        # Find cell label at nucleus centroid
        if 0 <= cy < cell_mask.shape[0] and 0 <= cx < cell_mask.shape[1]:
            cell_label = cell_mask[cy, cx]
            
            if cell_label > 0:
                # Get cell properties
                cell_props = measure.regionprops(cell_mask)
                for prop in cell_props:
                    if prop.label == cell_label:
                        return prop
        
        return None
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _fractal_dimension(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate fractal dimension (box-counting method)"""
        try:
            # Threshold image
            binary = image[mask] > np.mean(image[mask])
            
            if binary.sum() == 0:
                return 0.0
            
            # Minimal box-counting implementation
            # In practice, use more sophisticated methods
            def boxcount(Z, k):
                S = np.add.reduceat(
                    np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                    np.arange(0, Z.shape[1], k), axis=1
                )
                return len(np.where(S > 0)[0])
            
            # Reshape to square
            side = min(mask.shape[0], mask.shape[1])
            Z = np.zeros((side, side), dtype=bool)
            Z[:image[mask].reshape(-1).shape[0] % side] = binary.reshape(-1)[:side*side].reshape(side, side)
            
            # Count boxes at different scales
            scales = [2, 4, 8, 16, 32]
            Ns = []
            for scale in scales:
                if side >= scale:
                    Ns.append(boxcount(Z, scale))
            
            if len(Ns) < 2:
                return 0.0
            
            # Linear fit in log-log plot
            coeffs = np.polyfit(np.log(scales[:len(Ns)]), np.log(Ns), 1)
            return -coeffs[0]
            
        except:
            return 0.0


# === Example Usage ===

def main():
    """Example usage of WSI feature extractor"""
    parser = argparse.ArgumentParser()
    parser.add
    # Initialize extractor
    extractor = WSIFeatureExtractor(patch_size=224)
    
    # Process WSI
    features_df = extractor.process_wsi(
        wsi_path='path/to/slide.svs',
        nucleus_mask_path='path/to/nucleus_mask.png',
        cell_mask_path='path/to/cell_mask.png',  # Optional
        level=0,  # Highest resolution
        min_nucleus_area=20,
        max_nucleus_area=10000,
        output_path='path/to/output_features.csv'
    )
    
    print(f"\nExtracted features for {len(features_df)} cells")
    print(f"Feature dimensions: {features_df.shape}")
    print(f"\nFeature columns: {list(features_df.columns)}")
    
    # Show summary statistics
    print("\n=== Aging Marker Summary ===")
    aging_cols = [col for col in features_df.columns if 'aging_' in col]
    print(features_df[aging_cols].describe())
    
    print("\n=== Cell Type Distribution ===")
    celltype_cols = [col for col in features_df.columns if 'celltype_' in col]
    for col in celltype_cols:
        high_prob = (features_df[col] > 0.5).sum()
        print(f"{col}: {high_prob} cells ({100*high_prob/len(features_df):.1f}%)")
    
    return features_df


if __name__ == '__main__':
    # Example with synthetic data
    print("WSI Feature Extraction Pipeline")
    print("=" * 50)
    
    # Create synthetic example
    print("\nCreating synthetic example...")
    
    # Create small synthetic image and masks
    image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    # Create synthetic nucleus mask
    nucleus_mask = np.zeros((1000, 1000), dtype=np.int32)
    for i in range(10):
        center = (np.random.randint(100, 900), np.random.randint(100, 900))
        radius = np.random.randint(20, 40)
        cv2.circle(nucleus_mask, center, radius, i+1, -1)
    
    # Create synthetic cell mask (dilated nuclei)
    cell_mask = np.zeros_like(nucleus_mask)
    for label in range(1, 11):
        nucleus = (nucleus_mask == label)
        from scipy.ndimage import binary_dilation
        cell = binary_dilation(nucleus, iterations=15)
        cell_mask[cell] = label
    
    # Save temporary files
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, 'image.png')
        nucleus_path = os.path.join(tmpdir, 'nucleus.png')
        cell_path = os.path.join(tmpdir, 'cell.png')
        output_path = os.path.join(tmpdir, 'features.csv')
        
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(nucleus_path, nucleus_mask.astype(np.uint16))
        cv2.imwrite(cell_path, cell_mask.astype(np.uint16))
        
        # Run extractor
        extractor = WSIFeatureExtractor(use_deep_features=False)
        
        features_df = extractor.process_wsi(
            wsi_path=image_path,
            nucleus_mask_path=nucleus_path,
            cell_mask_path=cell_path,
            output_path=output_path
        )
        
        print(f"\n✓ Successfully extracted features for {len(features_df)} cells")
        print(f"✓ Feature dimensions: {features_df.shape}")
        print(f"\n✓ Sample of extracted features:")
        print(features_df.head())
        
        print(f"\n✓ Aging-specific features:")
        aging_features = [col for col in features_df.columns if 'aging_' in col]
        print(features_df[aging_features].describe())