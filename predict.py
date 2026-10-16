"""simple, explainable predictions with least-squares regression (numpy only)."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_CATEGORIES = 15  # text columns with more distinct values than this are skipped
ID_NAME = re.compile(r"(^|[_\s])(id|iso|code|num|number|zip|index)([_\s]|$)", re.IGNORECASE)
TIME_NAME = re.compile(r"^(year|month|day|date|time)$", re.IGNORECASE)


def looks_like_id(series):
    """ID numbers and codes say nothing about the values they label."""
    return bool(ID_NAME.search(str(series.name))) or (
        # Row numbers like 1, 2, 3, ... also count as IDs.
        pd.api.types.is_integer_dtype(series) and series.is_unique and len(series) > 20
        and series.max() - series.min() + 1 == len(series)
    )


def default_target(numeric_columns, df):
    """the first numeric column that looks like a measurement (not an ID or a year)."""
    for column in numeric_columns:
        if not looks_like_id(df[column]) and not TIME_NAME.match(str(column)):
            return column
    return numeric_columns[0]


def r_squared(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    total = ((actual - actual.mean()) ** 2).sum()
    return 1 - ((actual - predicted) ** 2).sum() / total if total else 0.0


def fit_quality(r2):
    if r2 >= 0.7:
        return "strong"
    if r2 >= 0.4:
        return "moderate"
    if r2 >= 0.15:
        return "weak"
    return "very weak"


# ---------------------------------------------------------------------------
# trend line between two columns
# ---------------------------------------------------------------------------

@dataclass
class Trend:
    coefficients: np.ndarray  # highest power first, as numpy.polyfit returns them
    r2: float
    x_is_date: bool

    def predict(self, x):
        x = np.asarray(x, float)
        return np.polyval(self.coefficients, x)

    def equation(self, x_name, y_name):
        terms = []
        degree = len(self.coefficients) - 1
        for power, coefficient in zip(range(degree, -1, -1), self.coefficients):
            name = "days" if self.x_is_date else x_name
            term = f"{abs(coefficient):.4g}" + (f" × {name}" if power == 1 else f" × {name}²" if power == 2 else "")
            terms.append(("− " if coefficient < 0 else "+ ") + term)
        text = " ".join(terms)
        return f"{y_name} = " + (text[2:] if text.startswith("+ ") else "−" + text[1:])


def to_numbers(series):
    """numbers as-is; dates become days since the first date."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.min()).dt.total_seconds() / 86400, True
    return pd.to_numeric(series, errors="coerce"), False


def fit_trend(df, x, y, degree=1):
    xs, is_date = to_numbers(df[x])
    ys = pd.to_numeric(df[y], errors="coerce")
    usable = xs.notna() & ys.notna()
    if usable.sum() < degree + 2:
        raise ValueError("Not enough rows with numbers in both columns to fit a trend.")
    if xs[usable].nunique() < 2:
        raise ValueError(f"'{x}' needs at least two different values.")
    coefficients = np.polyfit(xs[usable], ys[usable], degree)
    r2 = r_squared(ys[usable], np.polyval(coefficients, xs[usable]))
    return Trend(coefficients, r2, is_date), xs, ys


# ---------------------------------------------------------------------------
# what drives a target column
# ---------------------------------------------------------------------------

@dataclass
class DriverModel:
    target: str
    features: list
    design_columns: list  # columns of the encoded table, in coefficient order
    means: pd.Series
    stds: pd.Series
    coefficients: np.ndarray  # intercept first, then one per design column (standardized units)
    r2: float
    rows_used: int

    def importance(self):
        """one row per original feature: share of influence and direction."""
        weights = pd.Series(self.coefficients[1:], index=self.design_columns)
        rows = []
        for feature in self.features:
            parts = weights[[c for c in self.design_columns if c == feature or c.startswith(f"{feature} = ")]]
            if parts.empty:
                continue
            numeric = len(parts) == 1 and parts.index[0] == feature
            rows.append({
                "Column": feature,
                "Influence": float(parts.abs().sum()),
                "Direction": ("Higher → higher" if parts.iloc[0] > 0 else "Higher → lower") if numeric
                else "Depends on the category",
            })
        table = pd.DataFrame(rows).sort_values("Influence", ascending=False)
        total = table["Influence"].sum()
        table["Influence"] = table["Influence"] / total if total else 0.0
        return table.reset_index(drop=True)

    def predict_one(self, values):
        """predict the target for a dict of {feature: value}."""
        row = pd.DataFrame([values])
        encoded = encode(row, self.features).reindex(columns=self.design_columns, fill_value=0)
        scaled = (encoded - self.means) / self.stds
        return float(self.coefficients[0] + scaled.to_numpy(float)[0] @ self.coefficients[1:])


def usable_features(df, target):
    """columns that can be used to explain the target."""
    features = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 1 and not looks_like_id(series):
            features.append(column)
        elif not pd.api.types.is_numeric_dtype(series) and 1 < series.nunique() <= MAX_CATEGORIES:
            features.append(column)
    return features


def encode(df, features):
    """numbers stay as they are; each category of a text column becomes a 0/1 column."""
    parts = []
    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            parts.append(series.astype(float).rename(feature))
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=feature, prefix_sep=" = ", dtype=float)
            parts.append(dummies)
    return pd.concat(parts, axis=1)


def fit_drivers(df, target, features):
    if not features:
        raise ValueError("Pick at least one column that could explain the target.")
    data = df[[target] + features].dropna()
    if len(data) < len(features) + 5:
        raise ValueError("Not enough complete rows for this many columns. Pick fewer columns.")
    y = pd.to_numeric(data[target], errors="coerce").to_numpy(float)
    encoded = encode(data, features)
    # Drop one category per text column so the 0/1 columns aren't redundant.
    for feature in features:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            categories = [c for c in encoded.columns if c.startswith(f"{feature} = ")]
            encoded = encoded.drop(columns=categories[:1])
    encoded = encoded.loc[:, encoded.std() > 0]
    means, stds = encoded.mean(), encoded.std()
    scaled = ((encoded - means) / stds).to_numpy(float)
    design = np.column_stack([np.ones(len(scaled)), scaled])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    r2 = r_squared(y, design @ coefficients)
    return DriverModel(target, features, list(encoded.columns), means, stds, coefficients, r2, len(data))
