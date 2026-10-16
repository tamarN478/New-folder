"""data quality checks: find common problems in a dataset and suggest a one-click fix."""

from dataclasses import dataclass, field

import pandas as pd

# How many points each kind of problem costs a column's 0-100 score.
PENALTIES = {"high": 40, "medium": 20, "low": 10}


@dataclass
class Issue:
    column: str | None  # None for problems with the whole dataset
    severity: str  # "high", "medium" or "low"
    problem: str
    fix_label: str
    action: str  # a utils.clean_data action
    columns: list = field(default_factory=list)
    option: str | None = None


def _text_values(series):
    return series.dropna().astype(str)


def _is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def column_issues(df, column):
    """every problem found in one column."""
    series = df[column]
    issues = []
    rows = len(series)

    missing = int(series.isna().sum())
    if missing:
        share = missing / rows
        numeric = pd.api.types.is_numeric_dtype(series)
        issues.append(Issue(
            column, "high" if share > 0.3 else "medium" if share > 0.05 else "low",
            f"{missing:,} missing value{'s' if missing != 1 else ''} ({share:.0%})",
            "Fill with median" if numeric else "Fill with most common value",
            "Fill Missing Values", [column],
            "Median" if numeric else "Most Common Value (Mode)",
        ))

    if series.nunique(dropna=True) == 1 and rows > 1:
        issues.append(Issue(
            column, "low", "Every row has the same value, so it tells you nothing",
            "Drop column", "Drop Columns", [column],
        ))

    if _is_text(series) and series.notna().any():
        text = _text_values(series)
        as_numbers = pd.to_numeric(text, errors="coerce")
        numeric_share = as_numbers.notna().mean()
        if numeric_share == 1:
            issues.append(Issue(
                column, "medium", "Numbers are stored as text, so they can't be averaged or charted",
                "Convert to numbers", "Change Column Type", [column], "Number",
            ))
        elif numeric_share >= 0.6:
            bad = int(as_numbers.isna().sum())
            issues.append(Issue(
                column, "medium", f"Mostly numbers, but {bad:,} value{'s are' if bad != 1 else ' is'} text",
                "Convert to numbers (text becomes missing)", "Change Column Type", [column], "Number",
            ))

        padded = int((text != text.str.strip()).sum())
        if padded:
            issues.append(Issue(
                column, "low", f"{padded:,} value{'s have' if padded != 1 else ' has'} extra spaces",
                "Trim spaces", "Trim Spaces", [column],
            ))

        stripped = text.str.strip()
        variants = stripped.groupby(stripped.str.lower()).nunique()
        mixed_case = int((variants > 1).sum())
        if mixed_case:
            issues.append(Issue(
                column, "low",
                f"{mixed_case:,} value{'s are' if mixed_case != 1 else ' is'} written with different "
                "capitalization (e.g. 'math' and 'Math')",
                "Use Title Case", "Fix Capitalization", [column], "Title Case",
            ))

    if pd.api.types.is_numeric_dtype(series) and series.notna().sum() >= 8:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        spread = q3 - q1
        if spread > 0:
            outliers = int((~series.between(q1 - 1.5 * spread, q3 + 1.5 * spread) & series.notna()).sum())
            if outliers:
                issues.append(Issue(
                    column, "low", f"{outliers:,} unusually high or low value{'s' if outliers != 1 else ''}",
                    "Remove those rows", "Remove Outliers (IQR)", [column],
                ))
    return issues


def dataset_issues(df):
    """problems with the dataset as a whole."""
    issues = []
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(Issue(
            None, "medium" if duplicates / len(df) > 0.05 else "low",
            f"{duplicates:,} duplicate row{'s' if duplicates != 1 else ''}",
            "Remove duplicates", "Drop Duplicates",
        ))
    return issues


def column_score(issues):
    return max(0, 100 - sum(PENALTIES[issue.severity] for issue in issues))


def quality_report(df):
    """(overall score, per-column summary table, list of all issues)."""
    all_issues = dataset_issues(df)
    rows = []
    for column in df.columns:
        issues = column_issues(df, column)
        all_issues.extend(issues)
        rows.append({
            "Column": str(column),
            "Type": str(df[column].dtype),
            "Score": column_score(issues),
            "Problems": "; ".join(issue.problem for issue in issues) or "None found",
        })
    summary = pd.DataFrame(rows)
    overall = summary["Score"].mean() if len(summary) else 100
    overall -= sum(PENALTIES[i.severity] / 4 for i in all_issues if i.column is None)
    return max(0, round(overall)), summary, all_issues
