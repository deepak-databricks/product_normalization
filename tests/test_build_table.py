"""The table handed to Vector Search must be safe to index."""

import pytest
from pyspark.sql import functions as F

from product_normalization.ai_search.build_table import _drop_unusable, transform
from product_normalization.ai_search.embed_text import EMBED_TEXT_COL
from product_normalization.ai_search.ingest import COLUMNS, EXPECTED_ROWS
from product_normalization.config import ai_search_cfg, embedding_cols, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def built(load_fixture, cfg):
    """transform() applied to the clean rows of the sample CSV."""
    from product_normalization.ai_search.cleaning import has_field_shift

    raw = load_fixture("po_data_sample.csv").filter(~has_field_shift())
    return transform(raw, cfg)


def test_ingest_columns_match_the_real_csv_header():
    """Guards the hardcoded schema against drift from the actual file."""
    import csv
    import pathlib

    csv.field_size_limit(10**9)
    path = pathlib.Path(__file__).parent.parent / "assets" / "data" / "po_data_all.csv"
    if not path.is_file():
        pytest.skip("po_data_all.csv not present")
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == COLUMNS
    assert len(COLUMNS) == 31


def test_expected_rows_constant_is_documented():
    assert EXPECTED_ROWS == 1_999_911


def test_embed_text_is_present_and_populated(built):
    assert EMBED_TEXT_COL in built.columns
    assert built.filter(F.col(EMBED_TEXT_COL).isNull()).count() == 0


def test_output_keeps_id_and_label(built, cfg):
    flat = ai_search_cfg(cfg)
    assert flat["id_col"] in built.columns
    assert flat["label_col"] in built.columns


def test_output_keeps_every_embedding_col(built, cfg):
    """Results should be readable without joining back to the source table."""
    for col_name in embedding_cols(cfg):
        assert col_name in built.columns


def test_no_literal_sentinels_survive_in_embed_text(built):
    texts = [r[EMBED_TEXT_COL] for r in built.collect()]
    joined = " ".join(texts)
    for bad in (": None", ": NONE", ": nan", ": N/A", ": -"):
        assert bad not in joined


def test_domestic_rows_have_no_export_country_fragment(built):
    rows = built.filter(F.upper(F.col("IMPORT_DOMESTIC")) != "IMPORT").collect()
    assert rows, "sample should contain DOMESTIC rows"
    for row in rows:
        assert "export_country" not in row[EMBED_TEXT_COL]


def test_drop_unusable_removes_empty_rec_id(built, cfg):
    id_col = ai_search_cfg(cfg)["id_col"]
    before = built.count()
    after = _drop_unusable(built, id_col)
    # The sample seeds exactly one clean row with an empty REC_ID.
    assert after.count() == before - 1
    assert after.filter(F.trim(F.col(id_col)) == "").count() == 0


def test_unlabelled_rows_are_kept(built, cfg):
    """The cluster ID is eval ground truth, not an indexing prerequisite."""
    flat = ai_search_cfg(cfg)
    usable = _drop_unusable(built, flat["id_col"])
    unlabelled = usable.filter(F.trim(F.col(flat["label_col"])) == "")
    assert unlabelled.count() >= 1


def test_primary_key_is_unique_after_drop(built, cfg):
    """Vector Search rejects a non-unique primary key."""
    id_col = ai_search_cfg(cfg)["id_col"]
    usable = _drop_unusable(built, id_col)
    assert usable.count() == usable.select(id_col).distinct().count()
