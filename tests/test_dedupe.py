"""Collapsing to one row per embed_text must not lose or invent data."""

import pytest
from pyspark.sql import functions as F

from product_normalization.ai_search.build_table import transform
from product_normalization.ai_search.cleaning import has_field_shift
from product_normalization.ai_search.dedupe import (
    ROW_COUNT_COL,
    SOURCE_IDS_COL,
    dedupe,
)
from product_normalization.ai_search.embed_text import EMBED_TEXT_COL
from product_normalization.config import ai_search_cfg, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def full(load_fixture, cfg):
    """The sample CSV transformed the way build_table would leave it."""
    raw = load_fixture("po_data_sample.csv").filter(~has_field_shift())
    return transform(raw, cfg)


@pytest.fixture()
def duped(spark, full):
    """Force duplicate embed_text by unioning the sample with itself."""
    return full.unionByName(full)


def test_embed_text_is_unique_after_dedupe(duped, cfg):
    out = dedupe(duped, cfg)
    assert out.count() == out.select(EMBED_TEXT_COL).distinct().count()


def test_no_distinct_text_is_lost(duped, cfg):
    before = duped.select(EMBED_TEXT_COL).distinct().count()
    assert dedupe(duped, cfg).count() == before


def test_row_count_sums_back_to_the_source(duped, cfg):
    """The audit that proves nothing was dropped or double-counted."""
    out = dedupe(duped, cfg)
    total = out.agg(F.sum(ROW_COUNT_COL)).collect()[0][0]
    assert total == duped.count()


def test_duplicated_rows_collapse_to_count_two(duped, cfg):
    """Each text appears exactly twice in the self-union."""
    out = dedupe(duped, cfg)
    counts = {r[ROW_COUNT_COL] for r in out.collect()}
    assert counts == {2}


def test_representative_id_is_deterministic(duped, cfg):
    """min() keeps reruns stable, so the index primary key does not churn."""
    id_col = ai_search_cfg(cfg)["id_col"]
    first = {r[EMBED_TEXT_COL]: r[id_col] for r in dedupe(duped, cfg).collect()}
    second = {r[EMBED_TEXT_COL]: r[id_col] for r in dedupe(duped, cfg).collect()}
    assert first == second


def test_source_ids_are_retained_for_expansion(duped, cfg):
    """A match must be traceable back to the source rows."""
    out = dedupe(duped, cfg)
    for row in out.collect():
        assert row[SOURCE_IDS_COL], "source ids should never be empty"
        assert len(row[SOURCE_IDS_COL]) <= row[ROW_COUNT_COL]


def test_label_ambiguity_is_flagged_not_hidden(spark, full, cfg):
    """Two identical texts under different cluster IDs must be visible."""
    flat = ai_search_cfg(cfg)
    label_col = flat["label_col"]
    one = full.limit(1)
    conflicting = one.withColumn(label_col, F.lit("CONFLICTING_CLUSTER"))
    out = dedupe(one.unionByName(conflicting), cfg)
    assert out.filter(F.col("label_ambiguous")).count() == 1


def test_unambiguous_labels_are_not_flagged(duped, cfg):
    out = dedupe(duped, cfg)
    assert out.filter(F.col("label_ambiguous")).count() == 0


def test_embedding_cols_survive_the_collapse(duped, cfg):
    """Results stay readable without joining back to the full table."""
    out = dedupe(duped, cfg)
    for col in ai_search_cfg(cfg)["embedding_cols"]:
        assert col in out.columns
