"""Column-level fixes for known defects in the TJX PO extract.

All three defects are present in the files as the customer sent them; none is
introduced locally. See the ``data_quality`` block in config.yaml.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

# The escaping bug shifted fields rightward, so CONFIDENCE_SCORE ends up holding
# a fragment like '076"' instead of a number. Matches config.yaml's
# data_quality.malformed_row_filter.
NUMERIC_RE = r"^[0-9]+([.][0-9]+)?$"


def normalize_sentinels(col_name: str, sentinels: list[str]) -> Column:
    """Trim a column and map "missing" placeholders to real NULL.

    Matches the *whole* trimmed value only, never a substring, so a vendor named
    ``ST. JOHN`` survives even though ``.`` is a sentinel. Comparison is
    case-insensitive: the data carries both ``None`` and ``NONE``.
    """
    trimmed = F.trim(F.col(col_name))
    blanked = F.when(trimmed == F.lit(""), F.lit(None)).otherwise(trimmed)
    if not sentinels:
        return blanked
    lowered = [s.strip().lower() for s in sentinels if s.strip()]
    return F.when(F.lower(blanked).isin(lowered), F.lit(None)).otherwise(blanked)


def has_field_shift() -> Column:
    """True for rows broken by the double-escaped inch-mark bug.

    An inch mark in a product name (``12" CERAMIC WOK``) was double-escaped at
    export, which broke CSV parsing and pushed every later value one column
    right. Affected rows carry garbage in VENDOR_NAME and PRODUCT_DESCRIPTION --
    exactly the fields being embedded -- so they must not reach the index.
    Expect 3,176 of 1,999,911 rows (0.159%).
    """
    score = F.trim(F.col("CONFIDENCE_SCORE"))
    return ~(score.isNull() | (score == F.lit("")) | score.rlike(NUMERIC_RE))


def gate_export_country() -> Column:
    """NULL out EXPORT_COUNTRY unless the row is an IMPORT.

    EXPORT_COUNTRY is 64.41% missing, but structurally so: 0% missing on IMPORT
    rows, 59.8% on DOMESTIC, because a domestic purchase has no export country.
    Left as-is it leaks the import/domestic flag into the embedding, letting rows
    cluster on shipping mode instead of on what the product actually is.
    """
    return F.when(F.trim(F.upper(F.col("IMPORT_DOMESTIC"))) == F.lit("IMPORT"), F.col("EXPORT_COUNTRY")).otherwise(
        F.lit(None)
    )
