from .curl import compute_curl
from .features import extract_features
from .recommendations import (
    classify_hair_condition,
    format_output,
    generate_recommendations,
    prioritize_recommendations,
)
from .segmentation import segment_hair


def run_hair_pipeline(image):
    """Run segmentation, feature extraction, curl analysis, and recommendations."""
    hair_mask = segment_hair(image)
    features = extract_features(image, hair_mask)
    curl = compute_curl(image, hair_mask)
    state = classify_hair_condition(features, curl)
    recommendations = generate_recommendations(state, curl)
    plan = prioritize_recommendations(recommendations)

    return {
        "features": features,
        "curl": curl,
        "state": state,
        "recommendations": recommendations,
        "plan": plan,
        "report": format_output(state, plan),
    }
