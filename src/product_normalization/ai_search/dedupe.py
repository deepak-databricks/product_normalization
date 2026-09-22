"""Collapse the indexable table to one row per distinct ``embed_text``.

79.6% of rows share their embed_text with another row (1,996,496 rows -> 408,027
distinct texts; the worst text repeats 9,386 times), because the same product
recurs across POs and months with identical descriptive fields. Indexing all of
them embeds the same string thousands of times and lets exact duplicates crowd
the head of every result list, which is what pinned the first recall measurement
at a meaningless 1.0.

This builds a sibling table and index so the deduplicated approach can be
compared against the original rather than replacing it.
"""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import ai_search_cfg
from .embed_text import EMBED_TEXT_COL

logger = logging.getLogger(__name__)

# Columns that vary within a duplicate group and are therefore aggregated rather
# than picked arbitrarily.
ROW_COUNT_COL = "source_row_count"
SOURCE_IDS_COL = "source_rec_ids"
MAX_SOURCE_IDS = 20


def dedupe(df: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    """One row per distinct ``embed_text``.

    The surviving row keeps a representative ``REC_ID`` (the minimum, chosen for
    determinism so reruns are stable) and carries the group size plus a capped
    sample of member ids, so a match can be expanded back to every source row.

    Label handling: exactly 1 of 408,027 texts maps to more than one cluster
    (affecting 3 rows), so ``min`` is safe here. ``label_ambiguous`` flags that
    case rather than hiding it.
    """
    flat = ai_search_cfg(cfg)
    id_col, label_col = flat["id_col"], flat["label_col"]

    # Every column except the ones aggregated explicitly is identical within a
    # group by construction -- embed_text is built from them -- so first() is
    # exact, not an arbitrary pick.
    passthrough = [c for c in df.columns if c not in (id_col, label_col, EMBED_TEXT_COL)]

    aggs = [
        F.min(F.col(id_col)).alias(id_col),
        F.min(F.col(label_col)).alias(label_col),
        F.countDistinct(F.col(label_col)).alias("_nlabels"),
        F.count(F.lit(1)).alias(ROW_COUNT_COL),
        F.slice(F.sort_array(F.collect_set(F.col(id_col))), 1, MAX_SOURCE_IDS).alias(SOURCE_IDS_COL),
        *[F.first(F.col(c), ignorenulls=True).alias(c) for c in passthrough],
    ]

    return df.groupBy(EMBED_TEXT_COL).agg(*aggs).withColumn("label_ambiguous", F.col("_nlabels") > 1).drop("_nlabels")


def build_dedupe_table(
    spark: SparkSession,
    source_table: str,
    target_table: str,
    cfg: dict[str, Any],
) -> dict[str, int]:
    """Write the deduplicated sibling table. Returns counts for comparison."""
    flat = ai_search_cfg(cfg)
    id_col, label_col = flat["id_col"], flat["label_col"]

    source = spark.table(source_table)
    deduped = dedupe(source, cfg)

    deduped.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)
    spark.sql(f"ALTER TABLE {target_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    final = spark.table(target_table)
    stats = {
        "source_rows": source.count(),
        "dedupe_rows": final.count(),
        "distinct_ids": final.select(id_col).distinct().count(),
        "labelled": final.filter(F.trim(F.col(label_col)) != "").count(),
        "label_ambiguous": final.filter(F.col("label_ambiguous")).count(),
    }
    stats["collapse_ratio"] = round(stats["source_rows"] / max(stats["dedupe_rows"], 1), 2)

    if stats["dedupe_rows"] != stats["distinct_ids"]:
        raise ValueError(
            f"{id_col} is not unique in {target_table}: "
            f"{stats['dedupe_rows']:,} rows vs {stats['distinct_ids']:,} distinct ids"
        )

    logger.info("built %s: %s", target_table, stats)
    return stats
