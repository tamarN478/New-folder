"""ways to add new columns: formulas, date parts, and number ranges."""

import re

import pandas as pd

DATE_PARTS = {
    "Year": lambda d: d.dt.year,
    "Month": lambda d: d.dt.month,
    "Month name": lambda d: d.dt.month_name(),
    "Day": lambda d: d.dt.day,
    "Weekday": lambda d: d.dt.day_name(),
    "Quarter": lambda d: d.dt.quarter,
}


def _quote_columns(expression, columns):
    """wrap column names with spaces or symbols in backticks so pandas can read them."""
    for column in sorted(map(str, columns), key=len, reverse=True):
        if not column.isidentifier() and column in expression and f"`{column}`" not in expression:
            expression = expression.replace(column, f"`{column}`")
    return expression


def formula_column(df, expression):
    """evaluate a formula like 'price * quantity' and return the resulting values."""
    expression = expression.strip()
    if not expression:
        raise ValueError("Type a formula first.")
    if re.search(r"__|\bimport\b|\blambda\b|@", expression):
        raise ValueError("That formula uses something that isn't allowed.")
    try:
        result = df.eval(_quote_columns(expression, df.columns), engine="python")
    except SyntaxError:
        raise ValueError("The formula isn't complete or has a typo.") from None
    except pd.errors.UndefinedVariableError as error:
        raise ValueError(f"Unknown column in the formula: {error}") from None
    if not isinstance(result, pd.Series):
        # A plain number like "5" fills the whole column.
        result = pd.Series(result, index=df.index)
    return result


def add_formula_column(df, name, expression):
    new_df = df.copy()
    new_df[name] = formula_column(df, expression)
    return new_df


def add_date_parts(df, column, parts):
    dates = pd.to_datetime(df[column], errors="coerce", format="mixed")
    if dates.notna().sum() == 0:
        raise ValueError(f"'{column}' doesn't contain dates that can be read.")
    new_df = df.copy()
    position = new_df.columns.get_loc(column) + 1
    for offset, part in enumerate(parts):
        new_df.insert(position + offset, f"{column} {part.lower()}", DATE_PARTS[part](dates))
    return new_df


def add_ranges(df, column, name, bins):
    """sort a numeric column into ranges, e.g. 0-2, 2-3, 3-4.

    bins is either a number of equal-width ranges or a list of boundaries.
    """
    values = pd.to_numeric(df[column], errors="coerce")
    if isinstance(bins, int):
        low, high = values.min(), values.max()
        edges = [low + (high - low) * i / bins for i in range(bins + 1)]
    else:
        edges = list(bins)
    labels = [f"{round(a, 2):g} – {round(b, 2):g}" for a, b in zip(edges, edges[1:])]
    new_df = df.copy()
    new_df[name] = pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype(object)
    return new_df


def parse_boundaries(text):
    """turn '0, 2, 3, 4' into [0.0, 2.0, 3.0, 4.0]."""
    numbers = sorted({float(part) for part in re.split(r"[,\s;]+", text.strip()) if part})
    if len(numbers) < 2:
        raise ValueError("Enter at least two boundaries, e.g. 0, 2, 3, 4.")
    return numbers
