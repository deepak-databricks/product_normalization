"""config.yaml is the single source of truth for what gets embedded."""

import pytest

from product_normalization.config import (
    ai_search_cfg,
    embedding_cols,
    load_config,
    null_sentinels,
    require,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_ai_search_flattens_list_of_mappings(cfg):
    """ai_search is authored as a list of single-key mappings, not a mapping."""
    assert isinstance(cfg["ai_search"], list)
    flat = ai_search_cfg(cfg)
    assert {"embedding_cols", "id_col", "label_col", "blocking_cols"} <= set(flat)


def test_embedding_cols_order_is_preserved(cfg):
    """Order matters: it fixes the field order inside embed_text."""
    assert embedding_cols(cfg) == [
        "VENDOR_NAME",
        "VENDOR_STYLE",
        "PRODUCT_DESCRIPTION",
        "DEPARTMENT_DESCRIPTION",
        "CLASS_NAME",
        "CATEGORY_NAME",
        "BUYER_NAME",
        "EXPORT_COUNTRY",
    ]


def test_id_and_label_cols(cfg):
    """label_col is the customer-supplied cluster ID; id_col is the row key."""
    assert require(cfg, "id_col") == "REC_ID"
    assert require(cfg, "label_col") == "PS_PRODUCT_ID"


def test_label_col_is_not_an_embedding_col(cfg):
    """The cluster ID is ground truth for eval; embedding it would leak the answer."""
    assert require(cfg, "label_col") not in embedding_cols(cfg)


def test_id_col_is_not_an_embedding_col(cfg):
    """REC_ID is an opaque key with no semantic content."""
    assert require(cfg, "id_col") not in embedding_cols(cfg)


def test_null_sentinels_present(cfg):
    sentinels = null_sentinels(cfg)
    assert "None" in sentinels
    assert "NA" in sentinels


def test_flatten_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate key"):
        ai_search_cfg({"ai_search": [{"id_col": "A"}, {"id_col": "B"}]})


def test_flatten_rejects_non_mapping_entry():
    with pytest.raises(TypeError):
        ai_search_cfg({"ai_search": ["not_a_mapping"]})


def test_missing_section_raises():
    with pytest.raises(KeyError):
        ai_search_cfg({})


def test_require_reports_available_keys():
    with pytest.raises(KeyError, match="have:"):
        require({"ai_search": [{"id_col": "REC_ID"}]}, "nope")
