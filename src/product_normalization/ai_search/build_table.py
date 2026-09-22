"""Bronze -> the indexable table, plus a quarantine table for broken rows.

Vector Search reads the output table directly, so every row that reaches it must
have a usable primary key and something worth embedding.
"""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import ai_search_cfg, embedding_cols, filter_cols, null_sentinels
from .cleaning import gate_export_country, has_field_shift, normalize_sentinels
from .embed_text import EMBED_TEXT_COL, build_embed_text

logger = logging.getLogger(__name__)

ON_MALFORMED_CHOICES = ("quarantine", "drop", "keep")


GATED_SUFFIX = "_gated_for_embedding"


def transform(df: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    """Normalize sentinels, derive embed_text, keep filter columns queryable."""
    flat = ai_search_cfg(cfg)
    cols = embedding_cols(cfg)
    filters = filter_cols(cfg)
    sentinels = null_sentinels(cfg)

    # Normalize both roles: SOURCE_REGION carries literal "None" too, and a filter
    # on the string "None" would be a silent trap.
    for col_name in dict.fromkeys([*cols, *filters]):
        df = df.withColumn(col_name, normalize_sentinels(col_name, sentinels))

    # EXPORT_COUNTRY plays two roles with conflicting requirements. For EMBEDDING it
    # must be gated to IMPORT rows, or its structural nullity leaks the
    # import/domestic flag into similarity. For FILTERING it must stay raw: 538,918
    # DOMESTIC rows do carry a value, and gating the column itself would make them
    # unfilterable. So gate a throwaway copy, embed that, and leave the real column
    # untouched.
    embed_source = list(cols)
    gated = None
    if "EXPORT_COUNTRY" in cols:
        gated = "EXPORT_COUNTRY" + GATED_SUFFIX
        df = df.withColumn(gated, gate_export_country())
        embed_source = [gated if c == "EXPORT_COUNTRY" else c for c in cols]

    df = df.withColumn(EMBED_TEXT_COL, build_embed_text(embed_source))
    if gated:
        df = df.drop(gated)

    id_col = flat["id_col"]
    label_col = flat["label_col"]
    # filter_cols come from config so adding a filterable field needs no code edit.
    # EXPORT_COUNTRY appears in both lists; it is embedded (gated) *and* filterable,
    # and the dedupe below keeps a single copy.
    keep = [id_col, label_col, *cols, *filter_cols(cfg), EMBED_TEXT_COL]
    seen: set[str] = set()
    ordered = [c for c in keep if not (c in seen or seen.add(c))]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise ValueError(f"columns named in config are absent from the source: {missing}")
    return df.select(*ordered)


def _drop_unusable(df: DataFrame, id_col: str) -> DataFrame:
    """Rows without a primary key or without text cannot be indexed.

    Note this drops on *empty string* as well as NULL: 239 rows carry an empty
    REC_ID. Unlabelled rows are NOT dropped -- the cluster ID is only needed to
    evaluate matching, not to index a product.
    """
    key = F.trim(F.col(id_col))
    text = F.trim(F.col(EMBED_TEXT_COL))
    return df.filter(key.isNotNull() & (key != "") & text.isNotNull() & (text != ""))


def _create_table_with_cdf(spark: SparkSession, df: DataFrame, table: str) -> None:
    """Write the table with Change Data Feed enabled.

    Delta Sync indexes require CDF. Setting it via explicit ALTER after the write
    (rather than relying on a session default) means a schema-overwriting write
    cannot silently leave it off -- which would make the index stop updating with
    no error anywhere.
    """
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")


def build(
    spark: SparkSession,
    bronze_table: str,
    target_table: str,
    quarantine_table: str,
    cfg: dict[str, Any],
    on_malformed: str = "quarantine",
) -> dict[str, int]:
    """Produce the indexable table. Returns row counts for reconciliation."""
    if on_malformed not in ON_MALFORMED_CHOICES:
        raise ValueError(f"on_malformed must be one of {ON_MALFORMED_CHOICES}, got {on_malformed!r}")

    flat = ai_search_cfg(cfg)
    id_col = flat["id_col"]
    label_col = flat["label_col"]

    bronze = spark.table(bronze_table)
    shifted = bronze.filter(has_field_shift())
    sound = bronze.filter(~has_field_shift())

    stats: dict[str, int] = {"bronze": bronze.count()}

    if on_malformed == "keep":
        # Garbage in VENDOR_NAME/PRODUCT_DESCRIPTION will pollute similarity.
        logger.warning("on_malformed=keep: field-shifted rows will be indexed with corrupt values")
        source = bronze
        stats["quarantined"] = 0
    else:
        stats["malformed"] = shifted.count()
        source = sound
        if on_malformed == "quarantine":
            _create_table_with_cdf(spark, shifted, quarantine_table)
            stats["quarantined"] = stats["malformed"]
            logger.info("quarantined %s rows to %s", f"{stats['malformed']:,}", quarantine_table)
        else:
            stats["quarantined"] = 0
            logger.info("dropped %s malformed rows", f"{stats['malformed']:,}")

    transformed = transform(source, cfg)
    usable = _drop_unusable(transformed, id_col)
    stats["dropped_unusable"] = transformed.count() - usable.count()

    _create_table_with_cdf(spark, usable, target_table)

    final = spark.table(target_table)
    stats["indexable"] = final.count()
    stats["distinct_ids"] = final.select(id_col).distinct().count()
    stats["labelled"] = final.filter(F.trim(F.col(label_col)) != "").count()

    if stats["indexable"] != stats["distinct_ids"]:
        raise ValueError(
            f"{id_col} is not unique in {target_table}: "
            f"{stats['indexable']:,} rows vs {stats['distinct_ids']:,} distinct. "
            "Vector Search requires a unique primary key."
        )

    logger.info("built %s: %s", target_table, stats)
    return stats
