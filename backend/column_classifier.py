import pandas as pd
import numpy as np


# ================================
# HELPER FUNCTIONS
# ================================
def is_date_series(series):
    try:
        sample = series.dropna().astype(str)
        sample = sample.sample(min(20, len(sample)))
        parsed = pd.to_datetime(sample, errors='coerce', dayfirst=True)
        return parsed.notna().mean() > 0.8
    except:
        return False


def is_boolean(values):
    return values.issubset({"true","false","yes","no","0","1","y","n"})


def is_gender(values):
    return values.issubset({"male","female","m","f","0","1"})


# ================================
# SPLIT DATE DETECTION (NEW)
# ================================
def detect_split_date_columns(df):
    cols_lower = [c.lower() for c in df.columns]

    day_col = None
    month_col = None
    year_col = None

    for col in df.columns:
        c = col.lower()

        if "day" in c:
            day_col = col
        elif "month" in c:
            month_col = col
        elif "year" in c:
            year_col = col

    # At least year + month required
    if year_col and month_col:
        return True, {"day": day_col, "month": month_col, "year": year_col}

    return False, None


# ================================
# FEATURE EXTRACTION
# ================================
def extract_features(series):

    non_null = series.dropna()

    if len(non_null) == 0:
        return None

    unique = non_null.nunique()
    total = len(non_null)

    values = set(non_null.astype(str).str.lower().unique())

    return {
        "unique_ratio": unique / total,
        "unique_count": unique,
        "is_numeric": pd.api.types.is_numeric_dtype(series),
        "values": values,
        "min": None if not pd.api.types.is_numeric_dtype(series) else non_null.min(),
        "max": None if not pd.api.types.is_numeric_dtype(series) else non_null.max()
    }


# ================================
# RULE BASED CLASSIFICATION
# ================================
def rule_based_classification(col, features):

    col_lower = col.lower()

    if "id" in col_lower:
        return "id", 0.95

    if "name" in col_lower:
        return "name", 0.9

    if "count" in col_lower or "qty" in col_lower or "number" in col_lower:
        return "count", 0.9

    values = features["values"]

    if is_boolean(values):
        return "boolean", 0.95

    if is_gender(values):
        return "gender", 0.95

    if features["is_numeric"]:

        if features["unique_count"] <= 10 or features["unique_ratio"] < 0.05:
            return "category", 0.9

        if features["min"] is not None and features["max"] is not None:
            if features["min"] >= 0 and features["max"] < 10000:
                return "count", 0.8

        return "numeric", 0.9

    # Date detection
    if is_date_series(pd.Series(list(values))):
        return "date", 0.9

    if features["unique_count"] < 20:
        return "category", 0.7

    return "text", 0.6


# ================================
# ML-LIKE FALLBACK (HEURISTIC)
# ================================
def ml_fallback(features):

    scores = {
        "numeric": 1 if features["is_numeric"] else 0,
        "category": 1 - features["unique_ratio"],
        "text": features["unique_ratio"]
    }

    role = max(scores, key=scores.get)
    confidence = scores[role]

    return role, round(confidence, 2)


# ================================
# MAIN FUNCTION
# ================================
def detect_column_roles(df):

    roles = {}
    confidences = {}

    # 🔥 STEP 1: Detect split date columns first
    is_split_date, date_parts = detect_split_date_columns(df)

    if is_split_date:
        for col in df.columns:
            col_lower = col.lower()

            if (
                (date_parts["day"] and col == date_parts["day"]) or
                (date_parts["month"] and col == date_parts["month"]) or
                (date_parts["year"] and col == date_parts["year"])
            ):
                roles[col] = "date"
                confidences[col] = 0.95
                continue

    # 🔥 STEP 2: Normal column classification
    for col in df.columns:

        if col in roles:
            continue

        features = extract_features(df[col])

        if features is None:
            roles[col] = "unknown"
            confidences[col] = 0
            continue

        rule_role, rule_conf = rule_based_classification(col, features)
        ml_role, ml_conf = ml_fallback(features)

        if rule_conf >= 0.85:
            final_role = rule_role
            final_conf = rule_conf
        else:
            final_role = ml_role
            final_conf = ml_conf

        roles[col] = final_role
        confidences[col] = round(final_conf, 2)

    return roles, confidences