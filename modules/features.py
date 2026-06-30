import cv2
import numpy as np
from scipy.stats import circvar
from skimage.feature import canny

try:
    from .image_utils import normalize_mask, to_rgb_array
except ImportError:
    from image_utils import normalize_mask, to_rgb_array


EMPTY_FEATURES = {
    "edge_density": 0.0,
    "frizz_score": 0.0,
    "curl_variance": 0.0,
    "definition_score": 0.0,
    "shine_score": 0.0,
    "scalp_visibility": 0.0,
}


def extract_features(image, hair_mask):
    """
    Extracts numerical features from the isolated hair region of the image.
    
    Args:
        image: A PIL.Image.Image or a numpy.ndarray (RGB).
        hair_mask: A binary numpy array (0 and 1) representing the hair mask.
        
    Returns:
        A dictionary containing:
            - edge_density (float)
            - frizz_score (float)
            - curl_variance (float)
            - definition_score (float)
            - shine_score (float)
            - scalp_visibility (float)
    """
    image = to_rgb_array(image)
    hair_mask_resized = normalize_mask(hair_mask, image.shape)
    mask_sum = int(np.sum(hair_mask_resized))

    if mask_sum == 0:
        return EMPTY_FEATURES.copy()

    # Masked hair region extraction
    hair_region = image * np.expand_dims(hair_mask_resized, axis=-1)
    
    # Convert region to grayscale for texture/edge analysis
    gray = cv2.cvtColor(hair_region, cv2.COLOR_RGB2GRAY)
    
    # --- Frizz Proxy (Edge Density) ---
    edges = canny(gray, sigma=2)
    edge_density = float(np.sum(edges) / mask_sum)
    
    # --- Texture Analysis (Sobel Gradient Magnitude) ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    hair_strength = magnitude * hair_mask_resized
    frizz_score = float(np.sum(hair_strength) / mask_sum)
    
    # --- Curl Directional Variance ---
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    orientation = np.arctan2(dy, dx)
    
    # Compute circular variance only over mask pixels
    masked_orientation = orientation[hair_mask_resized == 1]
    if len(masked_orientation) > 0:
        curl_variance = float(circvar(
            masked_orientation,
            high=np.pi,
            low=-np.pi
        ))
    else:
        curl_variance = 0.0
        
    # --- Definition Score ---
    # Formula: 1/(1+edge_density) + 1/(1+frizz_score) + 1/(1+curl_variance)
    definition_score = float(
        1 / (1 + edge_density) +
        1 / (1 + frizz_score) +
        1 / (1 + curl_variance)
    )
    
    # --- Shine + Moisture Proxy ---
    hsv = cv2.cvtColor(hair_region.astype(np.uint8), cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2]
    shine_score = float(np.mean(brightness[hair_mask_resized == 1]))
    
    # --- Scalp Visibility Ratio ---
    # Convert original image to HSV to detect skin-like color range
    hsv_full = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_full, lower_skin, upper_skin)
    
    # Dilate hair mask to include scalp boundary regions
    kernel = np.ones((7, 7), np.uint8)
    expanded_hair = cv2.dilate(
        hair_mask_resized.astype(np.uint8),
        kernel,
        iterations=1
    )
    
    scalp_candidates = skin_mask * expanded_hair
    expanded_sum = np.sum(expanded_hair > 0)
    
    if expanded_sum > 0:
        scalp_visibility = float(np.sum(scalp_candidates > 0) / expanded_sum)
    else:
        scalp_visibility = 0.0
        
    return {
        "edge_density": edge_density,
        "frizz_score": frizz_score,
        "curl_variance": curl_variance,
        "definition_score": definition_score,
        "shine_score": shine_score,
        "scalp_visibility": scalp_visibility
    }
