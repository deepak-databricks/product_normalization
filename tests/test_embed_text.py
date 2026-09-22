"""embed_text is the only column Vector Search embeds, so its exact shape matters."""

import pytest
from pyspark.sql import functions as F

from product_normalization.ai_search.cleaning import gate_export_country, normalize_sentinels
from product_normalization.ai_search.embed_text import (
    EMBED_TEXT_COL,
    build_embed_text,
    field_prefix,
)
from product_normalization.config import embedding_cols, load_config, null_sentinels

COLS = ["VENDOR_NAME", "VENDOR_STYLE", "PRODUCT_DESCRIPTION"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def sample(spark, load_fixture, cfg):
    """The sample CSV with sentinels normalized and EXPORT_COUNTRY gated."""
    df = load_fixture("po_data_sample.csv")
    sentinels = null_sentinels(cfg)
    for col_name in embedding_cols(cfg):
        df = df.withColumn(col_name, normalize_sentinels(col_name, sentinels))
    return df.withColumn("EXPORT_COUNTRY", gate_export_country())


def _embed_for(df, rec_id, cols):
    row = df.filter(F.col("REC_ID") == rec_id).withColumn(EMBED_TEXT_COL, build_embed_text(cols)).collect()[0]
    return row[EMBED_TEXT_COL]


def test_field_prefix_is_derived_from_column_name():
    """No separate config key: the prefix is just the lowercased column name."""
    assert field_prefix("VENDOR_STYLE") == "vendor_style"
    assert field_prefix("PRODUCT_DESCRIPTION") == "product_description"


def test_prefixes_and_order(spark, sample):
    """Values appear prefixed, in the order embedding_cols lists them."""
    text = _embed_for(sample, "8590385112", COLS)
    assert text == (
        "vendor_name: RALPH LAUREN EUR/BLUE LABEL | "
        "vendor_style: 710849298007 | "
        "product_description: 601 POPLIN SPORT SHIRT"
    )


def test_reordering_embedding_cols_reorders_output(spark, sample):
    """embedding_cols alone controls the output -- no code change needed."""
    text = _embed_for(sample, "8590385112", ["VENDOR_STYLE", "VENDOR_NAME"])
    assert text == "vendor_style: 710849298007 | vendor_name: RALPH LAUREN EUR/BLUE LABEL"


def test_adding_a_column_changes_output(spark, sample):
    """Proves embed_text follows config rather than a hardcoded list."""
    short = _embed_for(sample, "8590385112", COLS)
    longer = _embed_for(sample, "8590385112", COLS + ["CLASS_NAME"])
    assert longer == short + " | class_name: L/S SHIRTS"


def test_null_field_leaves_no_empty_fragment(spark, sample):
    """A DOMESTIC row has no export country; the fragment must vanish entirely."""
    text = _embed_for(sample, "8590385114", COLS + ["EXPORT_COUNTRY"])
    assert "export_country" not in text
    assert not text.endswith("|")
    assert "||" not in text


def test_literal_none_never_reaches_embed_text(spark, sample):
    """'None' is a sentinel, not a value -- embedding it would cluster on noise."""
    text = _embed_for(sample, "8590385115", COLS + ["EXPORT_COUNTRY"])
    assert "None" not in text
    assert "export_country" not in text


def test_uppercase_none_is_also_a_sentinel(spark, sample):
    """VENDOR_NAME='NONE' and BUYER_NAME='None' both drop out."""
    text = _embed_for(sample, "8590385116", ["VENDOR_NAME", "VENDOR_STYLE", "BUYER_NAME"])
    assert "vendor_name" not in text
    assert "buyer_name" not in text
    assert text == "vendor_style: ZZ999"


def test_st_john_survives_sentinel_matching(spark, sample):
    """'.' is a sentinel, but only as a whole value -- 'ST. JOHN' is a real vendor."""
    text = _embed_for(sample, "8590385117", ["VENDOR_NAME"])
    assert text == "vendor_name: ST. JOHN"


def test_sparse_row_still_builds_from_primary_fields(spark, sample):
    """Row 12's secondary context is all sentinels; VSN + description carry it."""
    text = _embed_for(sample, "8590385120", embedding_cols(load_config()))
    assert "vendor_name: SPARSE VENDOR" in text
    assert "vendor_style: SP-9" in text
    assert "class_name" not in text
    assert "category_name" not in text
    assert "buyer_name" not in text


def test_embedded_newline_is_preserved(spark, sample):
    """The row with a newline inside its description round-trips intact."""
    text = _embed_for(sample, "8590385119", ["PRODUCT_DESCRIPTION"])
    assert "TWO LINE" in text and "DESCRIPTION HERE" in text


def test_empty_embedding_cols_raises():
    with pytest.raises(ValueError, match="nothing to embed"):
        build_embed_text([])
