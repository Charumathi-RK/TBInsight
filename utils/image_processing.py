import cv2
import numpy as np
from PIL import Image
import streamlit as st

class ImageProcessor:
    def __init__(self):
        """Initialize image processor with standard parameters."""
        self.target_size = (224, 224)
        self.normalization_range = (0, 1)
    
    def preprocess_image(self, image):
        """
        Preprocess chest X-ray image for TB detection model.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif len(image_array.shape) == 2:
                # Grayscale to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Resize image
            resized_image = cv2.resize(image_array, self.target_size)
            
            # Apply preprocessing pipeline
            processed_image = self._apply_preprocessing_pipeline(resized_image)
            
            return processed_image
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def _apply_preprocessing_pipeline(self, image):
        """Apply comprehensive preprocessing pipeline."""
        # 1. Noise reduction
        denoised = self._denoise_image(image)
        
        # 2. Contrast enhancement
        enhanced = self._enhance_contrast(denoised)
        
        # 3. Normalization
        normalized = self._normalize_image(enhanced)
        
        # 4. Lung region emphasis (optional enhancement)
        emphasized = self._emphasize_lung_regions(normalized)
        
        return emphasized
    
    def _denoise_image(self, image):
        """Remove noise from the image."""
        # Apply bilateral filter to reduce noise while preserving edges
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised
    
    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE."""
        # Convert to LAB color space for better contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _normalize_image(self, image):
        """Normalize image pixel values."""
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        return normalized
    
    def _emphasize_lung_regions(self, image):
        """Apply techniques to emphasize lung regions."""
        # This is a simplified lung region emphasis
        # In a production system, you might use more sophisticated lung segmentation
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply morphological operations to enhance lung structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Create a mask for lung regions (simplified approach)
        lung_mask = self._create_lung_mask(gray)
        
        # Apply mask to original image
        if len(image.shape) == 3:
            emphasized = image.copy()
            for i in range(3):
                emphasized[:, :, i] = emphasized[:, :, i] * (lung_mask / 255.0)
        else:
            emphasized = image * (lung_mask / 255.0)
        
        return emphasized
    
    def _create_lung_mask(self, gray_image):
        """Create a simplified lung mask."""
        # Apply threshold to separate lung regions
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes in lung regions
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask with largest contours (likely lung regions)
        mask = np.zeros_like(gray_image)
        if contours:
            # Sort contours by area and keep the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Draw the largest contours (up to 2 for lungs)
            for i, contour in enumerate(contours[:2]):
                if cv2.contourArea(contour) > 1000:  # Filter small areas
                    cv2.fillPoly(mask, [contour], (255,))
        
        return mask
    
    def validate_image(self, image):
        """Validate if the image is suitable for TB detection."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Check image dimensions
            height, width = image_array.shape[:2]
            
            if min(height, width) < 224:
                validation_results['warnings'].append(
                    f"Image resolution is low ({width}x{height}). Better results with higher resolution."
                )
            
            # Check if image is too small
            if min(height, width) < 100:
                validation_results['errors'].append("Image too small for reliable analysis")
                validation_results['is_valid'] = False
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                validation_results['warnings'].append(
                    "Unusual aspect ratio. Ensure image shows complete chest area."
                )
            
            # Check if image has sufficient contrast
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            contrast = np.std(gray)
            if contrast < 20:
                validation_results['warnings'].append(
                    "Low contrast image. Results may be less reliable."
                )
            
            # Check for potential medical imaging characteristics
            mean_intensity = np.mean(gray)
            if mean_intensity < 30 or mean_intensity > 200:
                validation_results['warnings'].append(
                    "Image brightness unusual for chest X-ray. Verify image type."
                )
            
        except Exception as e:
            validation_results['errors'].append(f"Image validation failed: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_image_statistics(self, image):
        """Get statistical information about the image."""
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            stats = {
                'shape': image_array.shape,
                'dtype': str(image_array.dtype),
                'size_mb': image_array.nbytes / (1024 * 1024),
                'min_value': float(np.min(image_array)),
                'max_value': float(np.max(image_array)),
                'mean_value': float(np.mean(image_array)),
                'std_value': float(np.std(image_array))
            }
            
            if len(image_array.shape) == 3:
                stats['channels'] = image_array.shape[2]
                stats['color_space'] = 'RGB' if image_array.shape[2] == 3 else 'RGBA'
            else:
                stats['channels'] = 1
                stats['color_space'] = 'Grayscale'
            
            return stats
            
        except Exception as e:
            return {'error': f"Could not analyze image: {str(e)}"}
