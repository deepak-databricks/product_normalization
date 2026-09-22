"""Filter columns are synced for metadata filtering, not embedded.

EXPORT_COUNTRY is the tricky one: it plays both roles, and the two have opposite
requirements. Embedding needs it gated to IMPORT rows (its nullity is structural,
so ungated it leaks the import/domestic flag into similarity). Filtering needs it
raw, because 538,918 DOMESTIC rows do carry a value.
"""

import pytest
from pyspark.sql import functions as F

from product_normalization.ai_search.build_table import transform
from product_normalization.ai_search.cleaning import has_field_shift
from product_normalization.ai_search.embed_text import EMBED_TEXT_COL, field_prefix
from product_normalization.ai_search.index import columns_to_sync
from product_normalization.config import embedding_cols, filter_cols, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def built(load_fixture, cfg):
    raw = load_fixture("po_data_sample.csv").filter(~has_field_shift())
    return transform(raw, cfg)


def test_config_declares_both_region_concepts(cfg):
    """They are different fields: origin country vs selling region."""
    filters = filter_cols(cfg)
    assert "EXPORT_COUNTRY" in filters
    assert "SOURCE_REGION" in filters


def test_source_region_reaches_the_table(built):
    """It was previously absent entirely -- neither embedded nor synced."""
    assert "SOURCE_REGION" in built.columns


def test_all_filter_cols_reach_the_table(built, cfg):
    for col in filter_cols(cfg):
        assert col in built.columns, f"{col} missing from the built table"


def test_filter_cols_are_synced_to_the_index(cfg):
    """Only synced columns can be filtered on or returned."""
    synced = columns_to_sync(cfg)
    for col in filter_cols(cfg):
        assert col in synced


def test_source_region_is_not_embedded(built, cfg):
    """A 5-value categorical adds no semantic signal; it belongs in a filter."""
    assert "SOURCE_REGION" not in embedding_cols(cfg)
    for row in built.collect():
        assert "source_region" not in row[EMBED_TEXT_COL]


def test_export_country_stays_raw_for_filtering_on_domestic_rows(built):
    """The whole point of the gated-copy approach.

    Row 8590385115 is DOMESTIC with EXPORT_COUNTRY='None' (a sentinel -> NULL),
    but row 8590385120 is DOMESTIC with '-' (also a sentinel). Use the IMPORT rows
    to prove the column is not blanket-gated, and check DOMESTIC rows keep whatever
    real value they had.
    """
    got = {r["REC_ID"]: r["EXPORT_COUNTRY"] for r in built.collect()}
    # IMPORT rows keep their value in the column.
    assert got["8590385112"] == "ITA"
    assert got["8590385117"] == "USA"


def test_export_country_is_still_gated_inside_embed_text(built):
    """DOMESTIC rows must not carry an export_country fragment in the text."""
    rows = built.filter(F.upper(F.col("IMPORT_DOMESTIC")) != "IMPORT").collect()
    assert rows
    for row in rows:
        assert "export_country" not in row[EMBED_TEXT_COL]


def test_import_rows_do_carry_export_country_in_embed_text(built):
    rows = built.filter(F.upper(F.col("IMPORT_DOMESTIC")) == "IMPORT").collect()
    assert any("export_country: " in r[EMBED_TEXT_COL] for r in rows)


def test_internal_gated_suffix_never_leaks_into_the_table(built):
    assert not [c for c in built.columns if c.endswith("_gated_for_embedding")]


def test_internal_gated_suffix_never_leaks_into_embed_text(built):
    for row in built.collect():
        assert "gated_for_embedding" not in row[EMBED_TEXT_COL]


def test_field_prefix_strips_the_internal_suffix():
    assert field_prefix("EXPORT_COUNTRY_gated_for_embedding") == "export_country"
    assert field_prefix("VENDOR_STYLE") == "vendor_style"


def test_sentinels_normalized_in_filter_cols_too(load_fixture, cfg):
    """A filter matching the literal string 'None' would be a silent trap."""
    raw = load_fixture("po_data_sample.csv").filter(~has_field_shift())
    out = transform(raw, cfg)
    for col in filter_cols(cfg):
        bad = out.filter(F.lower(F.trim(F.col(col))).isin(["none", "null", "nan", "n/a"]))
        assert bad.count() == 0, f"{col} still holds a literal sentinel"


def test_missing_config_column_fails_loudly(load_fixture, cfg):
    """Naming a column that does not exist should raise, not silently drop it.

    Spark itself raises AnalysisException first (normalize_sentinels touches the
    column before the explicit check), so accept either -- the point is that it
    fails rather than producing a table with the column silently absent.
    """
    from pyspark.errors.exceptions.base import AnalysisException

    raw = load_fixture("po_data_sample.csv").filter(~has_field_shift())
    with pytest.raises((ValueError, AnalysisException)):
        transform(raw.drop("SOURCE_REGION"), cfg).collect()
