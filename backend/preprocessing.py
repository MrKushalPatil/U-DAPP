import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ================================
# SMART DATE PARSER
# ================================
def smart_parse_date(series):

    def parse(x):
        x = str(x)

        try:
            if "/" in x:
                return pd.to_datetime(x, format="%m/%d/%y", errors='coerce')
            elif "-" in x:
                return pd.to_datetime(x, dayfirst=True, errors='coerce')
            else:
                return pd.to_datetime(x, errors='coerce')
        except:
            return pd.NaT

    return series.apply(parse)


# ================================
# BASIC CLEANING
# ================================
def basic_cleaning(df, roles):

    for col, role in roles.items():

        if col not in df.columns:
            continue

        if role == "count":
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        elif role == "numeric":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df[col] = np.clip(df[col], lower, upper)

        elif role == "category":
            df[col] = df[col].astype(str).fillna("unknown")

        elif role == "boolean":
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({
                "true": 1, "yes": 1, "y": 1, "1": 1,
                "false": 0, "no": 0, "n": 0, "0": 0
            })
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        elif role == "gender":
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({
                "m": "male", "male": "male", "1": "male",
                "f": "female", "female": "female", "0": "female"
            }).fillna("male")

        elif role == "text":
            df[col] = df[col].astype(str).str.strip().str.lower().replace("", "unknown")

        elif role == "date":
            df[col] = smart_parse_date(df[col])
            df[col] = df[col].fillna(pd.Timestamp("2000-01-01"))

    df = df.drop_duplicates()

    return df


# ================================
# APPLY FORMATS
# ================================
def apply_formats(df, roles, choices):

    for col, role in roles.items():

        if col not in df.columns:
            continue

        choice = choices.get(col, ("", "1"))[1]

        if role == "gender":
            if choice == "1":
                df[col] = df[col].replace({"male": "Male", "female": "Female"})
            elif choice == "2":
                df[col] = df[col].replace({"male": "M", "female": "F"})
            else:
                df[col] = df[col].replace({"male": 1, "female": 0})

        elif role == "boolean":

            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            if choice == "2":
                df[col] = df[col].replace({1: "True", 0: "False"})
            elif choice == "3":
                df[col] = df[col].replace({1: "Yes", 0: "No"})

        elif role == "date":
            if pd.api.types.is_datetime64_any_dtype(df[col]):

                if choice == "1":
                    df[col] = df[col].dt.strftime("%Y-%m-%d")

                elif choice == "5":
                    df[col + "_year"] = df[col].dt.year
                    df[col + "_month"] = df[col].dt.month
                    df[col + "_day"] = df[col].dt.day
                    df.drop(columns=[col], inplace=True)

    return df


# ================================
# PREPROCESSING (FIXED)
# ================================
def preprocess(df, roles, choices):

    scaler = MinMaxScaler()

    for col, role in list(roles.items()):

        if col not in df.columns:
            continue

        choice = choices.get(col, ("", "1"))[1]

        # skip identifiers
        if role in ["id", "name"]:
            continue

        # ============================
        # NUMERIC
        # ============================
        if role == "numeric":
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

            if choice == "1":
                df[[col]] = scaler.fit_transform(df[[col]])

        # ============================
        # CATEGORY (FIXED SAFE VERSION)
        # ============================
        elif role == "category":

            df[col] = df[col].astype(str).fillna("unknown")

            dummies = None  # IMPORTANT FIX

            if choice == "1":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)

            elif choice == "2":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(bool)

            else:
                # fallback safety
                dummies = pd.get_dummies(df[col], prefix=col)

            if dummies is not None:
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)

                # OPTIONAL: update roles safely (prevents UI mismatch bugs)
                roles.pop(col, None)
                for new_col in dummies.columns:
                    roles[new_col] = "category"

        # ============================
        # TEXT
        # ============================
        elif role == "text":
            df[col] = df[col].astype(str).str.lower().str.strip()

    return df