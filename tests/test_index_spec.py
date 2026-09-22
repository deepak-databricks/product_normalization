"""Guards on the Vector Search index spec.

The SDK calls ``.as_dict()`` on ``delta_sync_index_spec``, so passing a plain dict
raises ``AttributeError: 'dict' object has no attribute 'as_dict'`` -- and only at
runtime, inside the job, after ingest and transform have already succeeded. These
tests catch that locally instead.
"""

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

from product_normalization.ai_search.embed_text import EMBED_TEXT_COL
from product_normalization.ai_search.index import columns_to_sync, ensure_index
from product_normalization.config import ai_search_cfg, embedding_cols, load_config

INDEX = "users.akil_thomas.po_products_ai_search_index"
TABLE = "users.akil_thomas.po_products_ai_search"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def client():
    """A WorkspaceClient whose get_index reports "not found"."""
    fake = MagicMock()
    fake.vector_search_indexes.get_index.side_effect = Exception("does not exist")
    return fake


def _spec_from(client):
    return client.vector_search_indexes.create_index.call_args.kwargs["delta_sync_index_spec"]


def test_spec_is_a_typed_object_not_a_dict(client, cfg):
    """The regression: a dict here fails inside the SDK at runtime."""
    ensure_index(client, INDEX, "cfc_ai_search", TABLE, "databricks-gte-large-en", cfg)
    spec = _spec_from(client)
    assert isinstance(spec, DeltaSyncVectorIndexSpecRequest)
    assert hasattr(spec, "as_dict")
    spec.as_dict()  # must not raise


def test_embedding_source_column_is_typed(client, cfg):
    ensure_index(client, INDEX, "cfc_ai_search", TABLE, "databricks-gte-large-en", cfg)
    (column,) = _spec_from(client).embedding_source_columns
    assert isinstance(column, EmbeddingSourceColumn)
    assert column.name == EMBED_TEXT_COL
    assert column.embedding_model_endpoint_name == "databricks-gte-large-en"


def test_enums_are_used_for_type_and_pipeline(client, cfg):
    ensure_index(client, INDEX, "cfc_ai_search", TABLE, "databricks-gte-large-en", cfg)
    kwargs = client.vector_search_indexes.create_index.call_args.kwargs
    assert kwargs["index_type"] == VectorIndexType.DELTA_SYNC
    # TRIGGERED, not CONTINUOUS: the source is a static batch load.
    assert _spec_from(client).pipeline_type == PipelineType.TRIGGERED


def test_primary_key_comes_from_config(client, cfg):
    ensure_index(client, INDEX, "cfc_ai_search", TABLE, "databricks-gte-large-en", cfg)
    kwargs = client.vector_search_indexes.create_index.call_args.kwargs
    assert kwargs["primary_key"] == ai_search_cfg(cfg)["id_col"]


def test_existing_index_syncs_instead_of_recreating(cfg):
    """Reruns must not recreate the index -- that would re-embed ~2M rows."""
    fake = MagicMock()
    fake.vector_search_indexes.get_index.return_value = object()
    action = ensure_index(fake, INDEX, "cfc_ai_search", TABLE, "databricks-gte-large-en", cfg)
    assert action == "synced"
    fake.vector_search_indexes.sync_index.assert_called_once_with(INDEX)
    fake.vector_search_indexes.create_index.assert_not_called()


def test_columns_to_sync_covers_id_label_and_text(cfg):
    """Only synced columns come back from a query."""
    cols = columns_to_sync(cfg)
    flat = ai_search_cfg(cfg)
    assert flat["id_col"] in cols
    assert flat["label_col"] in cols
    assert EMBED_TEXT_COL in cols
    for col in embedding_cols(cfg):
        assert col in cols


def test_columns_to_sync_has_no_duplicates(cfg):
    cols = columns_to_sync(cfg)
    assert len(cols) == len(set(cols))


def test_columns_to_sync_excludes_struct_columns(cfg):
    """No struct is materialized; embed_text is the only derived column."""
    assert "embed_fields" not in columns_to_sync(cfg)
