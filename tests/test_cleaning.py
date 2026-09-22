"""Defect handling for the TJX extract: field shift, sentinels, structural nulls."""

import pytest
from pyspark.sql import functions as F

from product_normalization.ai_search.cleaning import (
    gate_export_country,
    has_field_shift,
    normalize_sentinels,
)
from product_normalization.config import load_config, null_sentinels


@pytest.fixture(scope="module")
def sentinels():
    return null_sentinels(load_config())


@pytest.fixture()
def raw(load_fixture):
    return load_fixture("po_data_sample.csv")


def test_field_shift_detects_inch_mark_rows(raw):
    """The 2 seeded rows carry '076\"' / '001\"' in CONFIDENCE_SCORE."""
    shifted = raw.filter(has_field_shift())
    assert shifted.count() == 2
    assert {r["CONFIDENCE_SCORE"] for r in shifted.collect()} == {'076"', '001"'}


def test_field_shift_leaves_clean_rows_alone(raw):
    clean = raw.filter(~has_field_shift())
    assert clean.count() == 10
    assert all(r["CONFIDENCE_SCORE"] == "1.0" for r in clean.collect())


def test_shifted_rows_are_the_ones_missing_class_num(raw):
    """Corroborates the detector: field shift also empties CLASS_NUM."""
    for row in raw.filter(has_field_shift()).collect():
        assert row["CLASS_NUM"] == ""


def test_sentinels_become_null(raw, sentinels):
    df = raw.withColumn("v", normalize_sentinels("EXPORT_COUNTRY", sentinels))
    got = {r["REC_ID"]: r["v"] for r in df.collect()}
    assert got["8590385115"] is None  # "None"
    assert got["8590385120"] is None  # "-"
    assert got["8590385114"] is None  # "" empty
    assert got["8590385112"] == "ITA"  # real value survives


def test_sentinel_matching_is_case_insensitive(raw, sentinels):
    """The data carries both 'None' and 'NONE'."""
    df = raw.withColumn("v", normalize_sentinels("VENDOR_NAME", sentinels))
    got = {r["REC_ID"]: r["v"] for r in df.collect()}
    assert got["8590385116"] is None  # "NONE"


def test_sentinel_matches_whole_value_only(raw, sentinels):
    """'.' is a sentinel, but 'ST. JOHN' must not be destroyed by it."""
    df = raw.withColumn("v", normalize_sentinels("VENDOR_NAME", sentinels))
    got = {r["REC_ID"]: r["v"] for r in df.collect()}
    assert got["8590385117"] == "ST. JOHN"


def test_export_country_gated_to_import_rows(raw):
    """Its 64% nullity is structural; ungated it leaks the import/domestic flag."""
    df = raw.withColumn("gated", gate_export_country())
    got = {r["REC_ID"]: (r["IMPORT_DOMESTIC"], r["gated"]) for r in df.collect()}

    assert got["8590385112"] == ("IMPORT", "ITA")
    assert got["8590385117"] == ("IMPORT", "USA")
    # DOMESTIC rows lose it even when a value was present.
    assert got["8590385115"][1] is None
    assert got["8590385120"][1] is None


def test_no_domestic_row_keeps_an_export_country(raw):
    df = raw.withColumn("gated", gate_export_country())
    leaked = df.filter((F.upper(F.col("IMPORT_DOMESTIC")) != "IMPORT") & F.col("gated").isNotNull())
    assert leaked.count() == 0
