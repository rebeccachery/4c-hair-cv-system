import cv2
import numpy as np
from scipy import fftpack

try:
    from .image_utils import normalize_mask, to_rgb_array
except ImportError:
    from image_utils import normalize_mask, to_rgb_array


EMPTY_CURL_RESULT = {
    "curl_embedding": {
        "curl_frequency": 0.0,
        "curl_curvature": 0.0,
        "orientation_variance": 0.0,
    },
    "curl_type_estimate": "Unknown (No Hair Detected)",
    "confidence": 0.0,
}


def compute_curl(image, hair_mask):
    """
    Computes curl geometry embeddings and approximates curl type (4A, 4B, 4C) and confidence.
    
    Args:
        image: A PIL.Image.Image or a numpy.ndarray (RGB).
        hair_mask: A binary numpy array (0 and 1) representing the hair mask.
        
    Returns:
        A dictionary containing:
            - curl_embedding (dict):
                - curl_frequency (float)
                - curl_curvature (float)
                - orientation_variance (float)
            - curl_type_estimate (str)
            - confidence (float)
    """
    image = to_rgb_array(image)
    hair_mask_resized = normalize_mask(hair_mask, image.shape)
    mask = hair_mask_resized.astype(bool)
    mask_sum = int(np.sum(mask))

    if mask_sum == 0:
        return {
            "curl_embedding": EMPTY_CURL_RESULT["curl_embedding"].copy(),
            "curl_type_estimate": EMPTY_CURL_RESULT["curl_type_estimate"],
            "confidence": EMPTY_CURL_RESULT["confidence"],
        }

    # Extract isolated hair region
    hair_region = image * np.expand_dims(mask, axis=-1)
    hair_region_gray = cv2.cvtColor(hair_region, cv2.COLOR_RGB2GRAY)
    
    # --- Local Orientation Field ---
    dx = cv2.Sobel(hair_region_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(hair_region_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    orientation = np.arctan2(dy, dx)
    orientation_masked = orientation[mask]
    
    # --- Curl Frequency Estimation via 2D Fast Fourier Transform (FFT) ---
    # Apply FFT on masked grayscale region to capture spatial curl periodicities
    f_coef = fftpack.fft2(hair_region_gray * mask)
    fshift = fftpack.fftshift(f_coef)
    magnitude_spectrum = np.abs(fshift)
    curl_frequency = float(np.mean(magnitude_spectrum))
    
    # --- Curvature Variance Estimation ---
    # Compute 1D gradient along the masked orientation array to represent angular changes
    if len(orientation_masked) > 1:
        curvature = np.gradient(orientation_masked)
        curl_curvature_score = float(np.var(curvature))
        orientation_var = float(np.var(orientation_masked))
    else:
        curl_curvature_score = 0.0
        orientation_var = 0.0
        
    # --- Curl Tightness Embedding ---
    curl_embedding = {
        "curl_frequency": curl_frequency,
        "curl_curvature": curl_curvature_score,
        "orientation_variance": orientation_var
    }
    
    # --- Approximate Curl Type Classification ---
    # Weighted geometry formula
    score = (
        curl_embedding["curl_frequency"] * 0.4 +
        curl_embedding["curl_curvature"] * 0.4 +
        curl_embedding["orientation_variance"] * 0.2
    )
    
    if score < 0.5:
        curl_type = "4A-like (looser curl structure)"
    elif score < 1.2:
        curl_type = "4B-like (defined curl structure)"
    else:
        curl_type = "4C-like (tight coil structure)"
        
    # --- Confidence Level Estimation ---
    # Low standard deviation between geometric features indicates stable curl structure
    stability = 1 / (1 + np.std([
        curl_embedding["curl_frequency"],
        curl_embedding["curl_curvature"],
        curl_embedding["orientation_variance"]
    ]))
    confidence = float(stability)
    
    return {
        "curl_embedding": curl_embedding,
        "curl_type_estimate": curl_type,
        "confidence": confidence
    }
