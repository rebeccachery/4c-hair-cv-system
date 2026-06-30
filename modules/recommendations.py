import json

def normalize_features(f):
    """
    Normalizes raw features to standard scale [0.0 - 1.0].
    
    Args:
        f (dict): Raw features dict from extract_features.
        
    Returns:
        dict: Normalized features.
    """
    # Safeguard against missing keys or invalid inputs
    edge_density = f.get("edge_density", 0.0)
    shine_score = f.get("shine_score", 0.0)
    definition_score = f.get("definition_score", 3.0)
    scalp_visibility = f.get("scalp_visibility", 0.0)
    
    return {
        "frizz_score": float(min(1.0, edge_density * 2)),
        "dryness_score": float(max(0.0, 1 - (shine_score / 120))),
        "definition_score": float(definition_score),
        "scalp_risk": float(scalp_visibility)
    }

def classify_hair_condition(features, curl):
    """
    Classifies the current hair condition based on features and curl parameters.
    Supports both raw and normalized features inputs for compatibility.
    
    Args:
        features (dict): Raw or normalized feature dictionary.
        curl (dict): Curl classification results.
        
    Returns:
        dict: Categorized conditions.
    """
    # Detect if raw features are passed and normalize them
    # Raw features contain keys like "edge_density" or "shine_score"
    if "edge_density" in features or "shine_score" in features:
        f = normalize_features(features)
    else:
        # Otherwise, assume already normalized features are passed
        f = features
        
    condition = {
        "moisture_state": None,
        "detangling_sensitivity": None,
        "maintenance_priority": None
    }
    
    # 1. Moisture Classification
    dryness = f.get("dryness_score", 0.0)
    if dryness > 0.6:
        condition["moisture_state"] = "dehydrated"
    elif dryness > 0.3:
        condition["moisture_state"] = "balanced-dry"
    else:
        condition["moisture_state"] = "moisturized"
        
    # 2. Manipulation Risk / Detangling Sensitivity
    # Use "4C" in curl_type to be robust against "4C-like" and "4C-like (tight coil structure)"
    curl_type = curl.get("curl_type_estimate", "") if curl else ""
    frizz = f.get("frizz_score", 0.0)
    
    if frizz > 0.7 and ("4C" in curl_type or "4C-like" in curl_type):
        condition["detangling_sensitivity"] = "high"
    elif frizz > 0.4:
        condition["detangling_sensitivity"] = "medium"
    else:
        condition["detangling_sensitivity"] = "low"
        
    # Backward compatibility: populate "manipulation_risk" to prevent KeyErrors
    condition["manipulation_risk"] = condition["detangling_sensitivity"]
    
    # 3. Maintenance Urgency / Priority
    scalp_risk = f.get("scalp_risk", 0.0)
    if scalp_risk > 0.2:
        condition["maintenance_priority"] = "protective-style-review-needed"
    elif dryness > 0.6:
        condition["maintenance_priority"] = "deep-moisture-intervention"
    else:
        condition["maintenance_priority"] = "routine-care"
        
    return condition

def generate_recommendations(condition, curl):
    """
    Generates actionable maintenance recommendations based on classified conditions.
    
    Args:
        condition (dict): Classified conditions dictionary.
        curl (dict): Curl classification results.
        
    Returns:
        list: Recommended maintenance action items.
    """
    recs = []
    
    # Moisture-based logic
    moisture = condition.get("moisture_state")
    if moisture == "dehydrated":
        recs.append("Deep condition within 48 hours")
        recs.append("Use leave-in conditioner + seal with oil/butter")
    elif moisture == "balanced-dry":
        recs.append("Light moisture refresh (water-based leave-in)")
        
    # Curl-based logic
    curl_type = curl.get("curl_type_estimate", "") if curl else ""
    if "4C" in curl_type:
        recs.append("Low-manipulation styling recommended")
        recs.append("Avoid frequent detangling without slip")
        
    # Frizz / structure / risk logic
    # Support both "manipulation_risk" and "detangling_sensitivity" for complete backward compatibility
    risk = condition.get("manipulation_risk") or condition.get("detangling_sensitivity")
    if risk == "high":
        recs.append("Protective styling recommended (twists, braids)")
        recs.append("Minimize heat + mechanical stress")
        
    # Scalp logic
    priority = condition.get("maintenance_priority")
    if priority == "protective-style-review-needed":
        recs.append("Check tension on protective style (edges + scalp comfort)")
        
    return recs

def prioritize_recommendations(recs):
    """
    Groups recommendations by urgency (urgent, short-term, long-term).
    
    Args:
        recs (list): List of recommendation strings.
        
    Returns:
        dict: Prioritized tasks.
    """
    return {
        "urgent": [r for r in recs if "Protective" in r or "48 hours" in r],
        "short_term": [r for r in recs if "leave-in" in r or "moisture" in r],
        "long_term": [r for r in recs if "Avoid" in r or "Minimize" in r]
    }

# Create alias to match Colab notebook namespace exactly
prioritize = prioritize_recommendations

def format_output(condition, plan):
    """
    Formats the analysis and action plan into a beautiful human-readable string.
    
    Args:
        condition (dict): Classified conditions dictionary.
        plan (dict): Prioritized action plan dictionary.
        
    Returns:
        str: Formatted report.
    """
    # Support both key namespaces
    risk = condition.get("manipulation_risk") or condition.get("detangling_sensitivity")
    
    return f"""
🧠 Hair State Summary
- Moisture State: {condition.get('moisture_state')}
- Risk Level: {risk}
- Care Priority: {condition.get('maintenance_priority')}

🚨 Urgent Actions:
{_format_list(plan.get('urgent', []))}

🧴 Short-Term Care:
{_format_list(plan.get('short_term', []))}

🧭 Long-Term Care:
{_format_list(plan.get('long_term', []))}
"""

def _format_list(items):
    if not items:
        return "  - None"
    return "\n".join(f"  - {item}" for item in items)
