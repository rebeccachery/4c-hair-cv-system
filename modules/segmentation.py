import numpy as np

try:
    from .image_utils import to_pil_image
except ImportError:
    from image_utils import to_pil_image

# Global cache for lazy loading the segmentation pipeline
_segmenter_cache = None

def get_segmenter():
    """
    Loads and caches the Hugging Face SegFormer model for image segmentation.
    Uses CUDA if available.
    """
    global _segmenter_cache
    if _segmenter_cache is None:
        import torch
        from transformers import pipeline

        # Detect CUDA GPU device; otherwise default to CPU (-1)
        device = 0 if torch.cuda.is_available() else -1
        _segmenter_cache = pipeline(
            "image-segmentation",
            model="nvidia/segformer-b0-finetuned-ade-512-512",
            device=device
        )
    return _segmenter_cache

def segment_hair(image):
    """
    Segment the hair region from an image.
    
    Args:
        image: A PIL.Image.Image, numpy.ndarray, or file path.
        
    Returns:
        A binary numpy array (np.uint8) of 0s and 1s representing the hair mask.
        Returns None if no hair label is detected.
    """
    image = to_pil_image(image)
    
    segmenter = get_segmenter()
    segments = segmenter(image)
    
    # Extract the mask corresponding to the hair label
    hair_mask_pil = None
    for seg in segments:
        if "hair" in seg['label'].lower():
            hair_mask_pil = seg['mask']
            break
            
    if hair_mask_pil is None:
        width, height = image.size
        return np.zeros((height, width), dtype=np.uint8)
        
    # Hugging Face segmentation mask is a PIL image with values 0 (background) and 255 (mask).
    # Convert this to a binary mask of 0s and 1s to prevent downstream overflow/errors.
    hair_mask = (np.array(hair_mask_pil) > 127).astype(np.uint8)
    return hair_mask
