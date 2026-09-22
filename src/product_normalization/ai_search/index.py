"""Create or sync the Vector Search index over the product table.

Managed embeddings are used, so Databricks embeds ``embed_text`` itself; there is
no embedding column to maintain in the source table.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

from ..config import ai_search_cfg, embedding_cols, filter_cols
from .embed_text import EMBED_TEXT_COL

logger = logging.getLogger(__name__)

ONLINE_STATES = {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}


def endpoint_state(client: WorkspaceClient, endpoint: str) -> str:
    detail = client.vector_search_endpoints.get_endpoint(endpoint)
    status = getattr(detail, "endpoint_status", None)
    state = getattr(status, "state", None)
    return str(getattr(state, "value", state) or "UNKNOWN")


def ensure_endpoint(client: WorkspaceClient, endpoint: str) -> None:
    """Fail early with a clear message if the endpoint is unusable.

    The endpoint is shared infrastructure created outside this project, so it can
    be resized or removed by others. Checking here turns an opaque create_index
    error into an actionable one.
    """
    try:
        state = endpoint_state(client, endpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Vector Search endpoint {endpoint!r} is not reachable: {exc}. Check the name and that it still exists."
        ) from exc

    if state not in ONLINE_STATES:
        raise RuntimeError(f"Vector Search endpoint {endpoint!r} is {state}, expected ONLINE")
    logger.info("endpoint %s is %s", endpoint, state)


def index_exists(client: WorkspaceClient, index_name: str) -> bool:
    try:
        client.vector_search_indexes.get_index(index_name)
        return True
    except Exception:  # noqa: BLE001 - SDK raises NotFound subclasses inconsistently
        return False


def columns_to_sync(cfg: dict[str, Any]) -> list[str]:
    """Columns retrievable from query results.

    Only synced columns come back from a query, so this includes the id, the
    cluster label (needed to score matches), the embedded text, and the source
    fields. ``embed_text`` is what gets embedded; the individual fields are here
    so results are readable without a join back to the table.
    """
    flat = ai_search_cfg(cfg)
    cols = [
        flat["id_col"],
        flat["label_col"],
        EMBED_TEXT_COL,
        *embedding_cols(cfg),
        # Synced so they can be used as query filters; only synced columns are
        # available to filter on or return.
        *filter_cols(cfg),
    ]
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def ensure_index(
    client: WorkspaceClient,
    index_name: str,
    endpoint: str,
    source_table: str,
    embedding_endpoint: str,
    cfg: dict[str, Any],
) -> str:
    """Create the index if absent, otherwise trigger a sync. Idempotent."""
    flat = ai_search_cfg(cfg)
    primary_key = flat["id_col"]

    if index_exists(client, index_name):
        logger.info("index %s exists; triggering sync", index_name)
        client.vector_search_indexes.sync_index(index_name)
        return "synced"

    logger.info("creating index %s on %s", index_name, endpoint)
    # The SDK calls .as_dict() on the spec, so these must be typed objects rather
    # than plain dicts -- a dict raises AttributeError inside create_index.
    client.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint,
        primary_key=primary_key,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name=EMBED_TEXT_COL,
                    embedding_model_endpoint_name=embedding_endpoint,
                )
            ],
            # The source is a static batch load, so CONTINUOUS would bill for
            # idle change-listening.
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=columns_to_sync(cfg),
        ),
    )
    return "created"


def wait_until_ready(client: WorkspaceClient, index_name: str, minutes: float = 0.0) -> bool:
    """Poll until the index reports ready. Returns False on timeout.

    Timing out is not an error: the initial embed of ~2M rows can outlast any
    reasonable job timeout, and the sync continues server-side regardless.
    """
    if minutes <= 0:
        return False

    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        detail = client.vector_search_indexes.get_index(index_name)
        status = getattr(detail, "status", None)
        ready = bool(getattr(status, "ready", False))
        message = getattr(status, "message", "") or ""
        if ready:
            logger.info("index %s is ready", index_name)
            return True
        logger.info("index %s not ready yet: %s", index_name, message)
        time.sleep(30)

    logger.warning(
        "index %s still syncing after %s minutes; it continues server-side. Check with: "
        "databricks vector-search-indexes get-index %s",
        index_name,
        minutes,
        index_name,
    )
    return False
