"""combine two datasets: join them side by side on a key, or stack one under the other."""

import pandas as pd

JOIN_TYPES = {
    "Keep all rows of my data": "left",
    "Only rows that match in both": "inner",
    "Keep all rows from both": "outer",
}


def _comparable(left, right):
    """make two key columns comparable; returns them plus a note if their types had to change."""
    if left.dtype == right.dtype:
        return left, right, None
    both_numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
    if both_numeric:
        return left.astype(float), right.astype(float), None
    return (left.astype(str).str.strip(), right.astype(str).str.strip(),
            f"'{left.name}' and '{right.name}' had different types, so they were compared as text.")


def join_datasets(left, right, left_keys, right_keys, how, right_name="other"):
    """join right onto left. Returns (result, stats dict, notes list)."""
    if len(left_keys) != len(right_keys) or not left_keys:
        raise ValueError("Pick the same number of matching columns on both sides.")
    left, right = left.copy(), right.copy()
    notes = []
    key_names = []
    for position, (lk, rk) in enumerate(zip(left_keys, right_keys)):
        key = f"__key{position}"
        left[key], right[key], note = _comparable(left[lk], right[rk])
        if note:
            notes.append(note)
        key_names.append(key)
    # The right-side key columns are duplicates of the left-side ones after the join.
    right = right.drop(columns=[rk for lk, rk in zip(left_keys, right_keys) if rk == lk or rk in left_keys])

    duplicated_keys = int(right.duplicated(subset=key_names).sum())
    if duplicated_keys:
        rows = "1 row" if duplicated_keys == 1 else f"{duplicated_keys:,} rows"
        verb = "shares" if duplicated_keys == 1 else "share"
        notes.append(f"{rows} in the second dataset {verb} a key with another row, "
                     "so some of your rows appear more than once.")

    result = left.merge(right, on=key_names, how=how, suffixes=("", f" ({right_name})"), indicator=True)
    # Rows that only exist in the second dataset still need their key values filled in.
    for lk, key in zip(left_keys, key_names):
        result[lk] = result[lk].where(result[lk].notna(), result[key])
    stats = {
        "matched": int((result["_merge"] == "both").sum()),
        "only_mine": int((result["_merge"] == "left_only").sum()),
        "only_other": int((result["_merge"] == "right_only").sum()),
    }
    result = result.drop(columns=key_names + ["_merge"]).reset_index(drop=True)
    return result, stats, notes


def suggest_keys(left, right):
    """columns with the same name in both datasets, best candidates first."""
    common = [c for c in left.columns if c in right.columns]
    return sorted(common, key=lambda c: -right[c].is_unique)


def stack_datasets(top, bottom, add_source=False, top_name="my data", bottom_name="other"):
    """put bottom's rows under top's. Returns (result, columns only in top, columns only in bottom)."""
    only_top = [c for c in top.columns if c not in bottom.columns]
    only_bottom = [c for c in bottom.columns if c not in top.columns]
    if add_source:
        top = top.assign(source=top_name)
        bottom = bottom.assign(source=bottom_name)
    result = pd.concat([top, bottom], ignore_index=True, sort=False)
    return result, only_top, only_bottom
