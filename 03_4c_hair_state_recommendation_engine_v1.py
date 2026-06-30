from modules.recommendations import (
    classify_hair_condition,
    format_output,
    generate_recommendations,
    prioritize_recommendations,
)


def main():
    hair_features = {
        "edge_density": 0.32,
        "curl_variance": 1.4,
        "definition_score": 0.45,
        "shine_score": 72,
        "scalp_visibility": 0.12,
    }
    curl_result = {
        "curl_type_estimate": "4C-like",
        "confidence": 0.78,
    }

    condition = classify_hair_condition(hair_features, curl_result)
    recommendations = generate_recommendations(condition, curl_result)
    plan = prioritize_recommendations(recommendations)
    print(format_output(condition, plan))


if __name__ == "__main__":
    main()
