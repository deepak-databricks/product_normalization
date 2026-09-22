"""Read the raw PO CSV from a UC volume into a bronze Delta table, unmodified.

The reader options here are load-bearing, not stylistic -- see ``read_options``.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T

logger = logging.getLogger(__name__)

# The unioned CSV has 31 columns; every one is read as STRING deliberately.
COLUMNS = [
    "IMPORT_DOMESTIC",
    "CHANNEL",
    "PO_PRE_NUM",
    "PO_MSTR_NUM",
    "SKU",
    "VENDOR_NAME",
    "VENDOR_STYLE",
    "PRODUCT_DESCRIPTION",
    "DEPARTMENT_NUM",
    "DEPARTMENT_DESCRIPTION",
    "LP_YYYY_MM",
    "TOTAL_CURRENT_UNITS",
    "TOTAL_ORIGINAL_UNITS",
    "TOTAL_ORDERED_UNITS",
    "EXPORT_COUNTRY",
    "BUYER_NAME",
    "BANNER",
    "VENDOR_NUMBER",
    "MASTER_VENDOR_NUMBER",
    "CON_CAN_DATE",
    "CLASS_NAME",
    "CATEGORY_NAME",
    "PO_CREATE_DATE",
    "SOURCE_REGION",
    "DC_NUM",
    "BUYER_CODE",
    "REC_ID",
    "CONFIDENCE_SCORE",
    "CLASS_NUM",
    "CATEGORY_NUM",
    "PS_PRODUCT_ID",
]

# Row count of assets/data/po_data_all.csv. Asserting on it is the cheapest
# possible guard against a silent CSV-parsing regression.
EXPECTED_ROWS = 1_999_911


def schema() -> T.StructType:
    """All-STRING schema.

    Never use ``inferSchema``: it would sample the file and coerce DEPARTMENT_NUM,
    CLASS_NUM and CATEGORY_NUM to integers, destroying leading zeros ('029' ->
    29). Those columns are categorical codes, not numbers.
    """
    return T.StructType([T.StructField(name, T.StringType(), nullable=True) for name in COLUMNS])


def read_options() -> dict[str, str]:
    """CSV options required by this specific export.

    - ``multiLine``: 14 of the 25 source parts contain newlines *inside* quoted
      fields. Without this, those records split across rows and the parse is
      silently wrong.
    - ``escape='"'``: the export double-escapes quotes (``\\"\\"``). Spark's
      default escape is backslash, which mis-parses them and changes how many
      rows look field-shifted.
    """
    return {
        "header": "true",
        "multiLine": "true",
        "quote": '"',
        "escape": '"',
        "mode": "PERMISSIVE",
    }


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.options(**read_options()).schema(schema()).csv(path)


def ingest(
    spark: SparkSession,
    source_path: str,
    bronze_table: str,
    expected_rows: int | None = EXPECTED_ROWS,
) -> int:
    """Load the CSV verbatim into ``bronze_table``. Returns the row count."""
    logger.info("reading %s", source_path)
    df = read_csv(spark, source_path)

    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table)
    count = spark.table(bronze_table).count()
    logger.info("wrote %s rows to %s", f"{count:,}", bronze_table)

    if expected_rows is not None and count != expected_rows:
        raise ValueError(
            f"{bronze_table} has {count:,} rows, expected {expected_rows:,}. "
            "A mismatch usually means the CSV reader options changed -- check that "
            "multiLine and escape are still set (see read_options)."
        )
    return count
