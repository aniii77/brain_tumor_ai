import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io

class ImageProcessor:
    """Image processing utilities for medical images"""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.supported_formats = ['PNG', 'JPEG', 'JPG', 'BMP', 'TIFF']
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for model input"""
        try:
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to RGB if necessary
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # Remove alpha channel if present
                image_array = image_array[:, :, :3]
            elif len(image_array.shape) == 2:
                # Convert grayscale to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Resize to target size
            processed_image = cv2.resize(image_array, self.target_size)
            
            # Normalize pixel values
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Apply medical image enhancements
            enhanced_image = self.enhance_medical_image(processed_image)
            
            return enhanced_image
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def enhance_medical_image(self, image_array):
        """Apply medical image specific enhancements"""
        try:
            # Convert back to uint8 for OpenCV operations
            uint8_image = (image_array * 255).astype(np.uint8)
            
            # Convert to grayscale for processing
            if len(uint8_image.shape) == 3:
                gray = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = uint8_image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to RGB
            if len(image_array.shape) == 3:
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            else:
                enhanced_rgb = enhanced_gray
            
            # Normalize back to 0-1 range
            enhanced_image = enhanced_rgb.astype(np.float32) / 255.0
            
            return enhanced_image
            
        except Exception as e:
            # Return original image if enhancement fails
            return image_array
    
    def validate_image(self, image):
        """Validate uploaded image"""
        try:
            if not isinstance(image, Image.Image):
                return False, "Invalid image format"
            
            if image.format not in self.supported_formats:
                return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
            
            # Check image dimensions
            width, height = image.size
            if width < 50 or height < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            
            if width > 2048 or height > 2048:
                return False, "Image too large (maximum 2048x2048 pixels)"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def extract_metadata(self, image):
        """Extract metadata from medical image"""
        metadata = {}
        
        try:
            if isinstance(image, Image.Image):
                metadata['format'] = image.format
                metadata['size'] = image.size
                metadata['mode'] = image.mode
                
                # Extract EXIF data if available
                if hasattr(image, '_getexif') and image._getexif() is not None:
                    exif_data = image._getexif()
                    metadata['exif'] = exif_data
                
                # Calculate basic statistics
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    # Convert to grayscale for statistics
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array
                
                metadata['statistics'] = {
                    'mean': float(np.mean(gray)),
                    'std': float(np.std(gray)),
                    'min': float(np.min(gray)),
                    'max': float(np.max(gray))
                }
                
        except Exception as e:
            metadata['error'] = f"Error extracting metadata: {str(e)}"
        
        return metadata
    
    def detect_image_quality(self, image_array):
        """Detect image quality metrics"""
        quality_metrics = {}
        
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor((image_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (image_array * 255).astype(np.uint8)
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = float(laplacian_var)
            
            # Contrast measurement
            contrast = np.std(gray)
            quality_metrics['contrast'] = float(contrast)
            
            # Brightness measurement
            brightness = np.mean(gray)
            quality_metrics['brightness'] = float(brightness)
            
            # Noise estimation (using local standard deviation)
            noise_filter = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.std(gray - noise_filter)
            quality_metrics['noise_level'] = float(noise)
            
            # Overall quality score (0-1)
            # Good sharpness: > 100, Good contrast: > 50, Good brightness: 50-200
            sharpness_score = min(laplacian_var / 100, 1.0)
            contrast_score = min(contrast / 50, 1.0)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            noise_score = max(0, 1.0 - noise / 30)
            
            overall_quality = (sharpness_score + contrast_score + brightness_score + noise_score) / 4
            quality_metrics['overall_quality'] = float(overall_quality)
            
        except Exception as e:
            quality_metrics['error'] = f"Error detecting quality: {str(e)}"
        
        return quality_metrics
    
    def apply_medical_filters(self, image_array):
        """Apply medical imaging specific filters"""
        try:
            # Convert to uint8 for OpenCV operations
            uint8_image = (image_array * 255).astype(np.uint8)
            
            if len(uint8_image.shape) == 3:
                gray = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = uint8_image
            
            # Apply different medical imaging filters
            filters = {}
            
            # Edge enhancement
            kernel_edge = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            edge_enhanced = cv2.filter2D(gray, -1, kernel_edge)
            filters['edge_enhanced'] = edge_enhanced
            
            # Gaussian blur for noise reduction
            gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            filters['gaussian_blur'] = gaussian_blur
            
            # Median filter for salt and pepper noise
            median_filter = cv2.medianBlur(gray, 5)
            filters['median_filter'] = median_filter
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            filters['opening'] = opening
            filters['closing'] = closing
            
            return filters
            
        except Exception as e:
            return {'error': f"Error applying filters: {str(e)}"}
    
    def create_image_thumbnail(self, image, size=(128, 128)):
        """Create thumbnail of the image"""
        try:
            if isinstance(image, Image.Image):
                thumbnail = image.copy()
                thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
                return thumbnail
            else:
                # Convert numpy array to PIL Image
                if len(image.shape) == 3:
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                else:
                    pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
                
                thumbnail = pil_image.copy()
                thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
                return thumbnail
                
        except Exception as e:
            return None
