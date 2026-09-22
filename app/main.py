"""FastAPI web app for searching and evaluating Databricks Vector Search index over TJX PO data."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

# Load .env before the module-level os.environ reads below (they run at import).
# python-dotenv ships in the `app` optional-dependency group; degrade gracefully
# if it or the .env file is absent so the app still runs on plain env vars.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from databricks.sdk import WorkspaceClient
from databricks.sql import connect as sql_connect
from databricks.sql.client import Connection
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from product_normalization.ai_search.dedupe import ROW_COUNT_COL
from product_normalization.ai_search.embed_text import EMBED_TEXT_COL
from product_normalization.ai_search.evaluate import OVERFETCH, query_index, score_batch
from product_normalization.ai_search.index import columns_to_sync, endpoint_state
from product_normalization.config import ai_search_cfg, embedding_cols, filter_cols, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals initialized at startup
_config = None
_client = None
_sql_connection = None
_filter_values_cache = None
_catalog = os.environ.get("DATABRICKS_CATALOG", "users")
_schema = os.environ.get("DATABRICKS_SCHEMA", "akil_thomas")
WAREHOUSE_HTTP_PATH = os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH", "/sql/1.0/warehouses/148ccb90800933a1")
# Distinct-text count per cluster, carried on sampled rows to give precision its
# denominator. Underscore-free so it survives a SELECT t.*, e.<col> round trip.
SIBLINGS_KEY = "cluster_distinct_texts"


def _jsonable(value: Any) -> Any:
    """Coerce a SQL cell into something FastAPI can serialize.

    ARRAY columns (the dedupe table's source_rec_ids) arrive as numpy ndarrays,
    which jsonable_encoder cannot handle -- it fails the whole response.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _row_to_dict(col_names: list[str], row: Any) -> dict[str, Any]:
    return {name: _jsonable(val) for name, val in zip(col_names, row)}


def get_sql_credentials_provider():
    """Adapt the SDK's auth to what databricks-sql-connector expects.

    The AZURE-SA-WORKSPACE profile authenticates as `databricks-cli`, so
    `config.token` is empty -- passing auth_type="pat" with a None token fails on
    the first connect. `config.authenticate` returns the request headers for
    whatever auth the profile actually uses, which is what the connector wants
    from a credentials_provider.
    """
    config = get_client().config
    return lambda: config.authenticate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle — setup and teardown resources."""
    # Startup: initialize resources
    logger.info("Starting up FastAPI app")
    yield
    # Shutdown: cleanup
    logger.info("Shutting down FastAPI app")
    global _sql_connection
    if _sql_connection:
        try:
            _sql_connection.close()
        except Exception as e:
            logger.warning(f"Error closing SQL connection: {e}")
        _sql_connection = None


app = FastAPI(
    title="TJX PO Search",
    description="Vector Search evaluation tool for product matching",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_config() -> dict[str, Any]:
    """Get or load configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_client() -> WorkspaceClient:
    """Get or create Databricks client."""
    global _client
    if _client is None:
        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "AZURE-SA-WORKSPACE")
        logger.info(f"Creating Databricks client with profile {profile}")
        _client = WorkspaceClient(profile=profile)
    return _client


def get_sql_connection() -> Connection:
    """Get or create a persistent SQL connection via databricks-sql-connector."""
    global _sql_connection
    if _sql_connection is None:
        config = get_client().config
        # server_hostname wants a bare host; config.host carries the https:// scheme.
        hostname = urlparse(config.host).hostname
        try:
            logger.info(f"Creating SQL connection to {hostname} via {WAREHOUSE_HTTP_PATH}")
            _sql_connection = sql_connect(
                server_hostname=hostname,
                http_path=WAREHOUSE_HTTP_PATH,
                credentials_provider=get_sql_credentials_provider(),
            )
        except Exception as e:
            logger.error(f"Failed to create SQL connection: {e}")
            raise

    return _sql_connection


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _get_filter_values(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Fetch distinct values for each filter column from the database."""
    global _filter_values_cache
    if _filter_values_cache is not None:
        return _filter_values_cache

    filter_col_names = filter_cols(cfg)
    filter_values = {}

    try:
        conn = get_sql_connection()
        table = f"{_catalog}.{_schema}.po_products_ai_search"

        for col in filter_col_names:
            try:
                # Fetch up to 200 distinct values, sorted
                query = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY 1 LIMIT 200"
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                values = [str(row[0]) for row in rows if row[0] is not None]
                filter_values[col] = values
                logger.info(f"Fetched {len(values)} distinct values for {col}")
            except Exception as e:
                logger.warning(f"Failed to fetch values for {col}: {e}")
                filter_values[col] = []

    except Exception as e:
        logger.warning(f"Failed to fetch filter values: {e}")
        # Return empty dict structure
        filter_values = {col: [] for col in filter_col_names}

    _filter_values_cache = filter_values
    return filter_values


@app.get("/api/config")
async def get_api_config():
    """Return configuration for the UI: embedding columns, filter columns, their values, and index names."""
    cfg = get_config()
    client = get_client()
    flat = ai_search_cfg(cfg)

    embed_cols = embedding_cols(cfg)
    filter_col_names = filter_cols(cfg)

    # Get distinct values for filter columns from the database
    filter_values = _get_filter_values(cfg)

    # Get index names and endpoint state
    dedupe_index = f"{_catalog}.{_schema}.po_products_ai_search_dedupe_index"
    full_index = f"{_catalog}.{_schema}.po_products_ai_search_index"

    try:
        endpoint_name = flat.get("endpoint", "cfc_ai_search")
        dedupe_state = endpoint_state(client, endpoint_name)
    except Exception as e:
        logger.warning(f"Failed to get endpoint state: {e}")
        dedupe_state = "UNKNOWN"

    return {
        "embedding_cols": embed_cols,
        "filter_cols": filter_col_names,
        "filter_values": filter_values,
        "dedupe_index": dedupe_index,
        "full_index": full_index,
        "endpoint_state": dedupe_state,
    }


@app.get("/api/samples")
async def get_samples(
    n: int = Query(5, ge=1, le=100),
    multi_vsn_only: bool = Query(False),
    vendor_style: str | None = Query(None, description="Match VENDOR_STYLE (substring, case-insensitive)"),
    vendor_name: str | None = Query(None, description="Match VENDOR_NAME (substring, case-insensitive)"),
):
    """Return n sample records from the dedupe table using SQL.

    With ``vendor_style`` (and/or ``vendor_name``) this becomes a lookup rather
    than a random draw: find the specific product you want to query the index
    with. Matching is a case-insensitive substring so a root style finds its
    suffixed variants -- searching M15371 also returns M15371-7395, which is
    exactly the hard-case family worth inspecting.

    The eligibility filter is relaxed for a targeted lookup. A cluster with only
    one distinct text cannot be *scored*, but you should still be able to find
    and search with the product.
    """
    cfg = get_config()
    flat = ai_search_cfg(cfg)

    dedupe_table = f"{_catalog}.{_schema}.po_products_ai_search_dedupe"
    label_col = flat["label_col"]
    id_col = flat["id_col"]

    samples = []
    try:
        conn = get_sql_connection()
        targeted = bool((vendor_style or "").strip() or (vendor_name or "").strip())

        # Parameter markers, never interpolation -- a vendor name like LEVI'S
        # would otherwise break the predicate, and user input must not reach SQL
        # as text.
        params: list[str] = []
        where = [f"TRIM({label_col}) != ''"]
        if (vendor_style or "").strip():
            where.append("LOWER(VENDOR_STYLE) LIKE ?")
            params.append(f"%{vendor_style.strip().lower()}%")
        if (vendor_name or "").strip():
            where.append("LOWER(VENDOR_NAME) LIKE ?")
            params.append(f"%{vendor_name.strip().lower()}%")
        where_sql = " AND ".join(where)

        if targeted:
            # Ordered so the most useful rows surface first: biggest clusters,
            # then the products that absorbed the most PO rows. Deterministic, so
            # the same search always returns the same list.
            sample_query = f"""
                WITH scored AS (
                    SELECT d.*,
                           COUNT(*) OVER (PARTITION BY d.{label_col}) AS cluster_distinct_texts
                    FROM {dedupe_table} d
                    WHERE {where_sql}
                )
                SELECT *
                FROM scored
                ORDER BY cluster_distinct_texts DESC, {ROW_COUNT_COL} DESC, VENDOR_STYLE
                LIMIT {n}
            """
        else:
            vsn_check = (
                "COUNT(DISTINCT CONCAT_WS('|', VENDOR_NAME, VENDOR_STYLE)) > 1" if multi_vsn_only else "1=1"
            )
            # Keep the cluster set server-side. Materializing every eligible id and
            # inlining it into an IN list builds a multi-megabyte query -- there are
            # tens of thousands of eligible clusters.
            sample_query = f"""
                WITH eligible AS (
                    SELECT {label_col}
                    FROM {dedupe_table}
                    WHERE TRIM({label_col}) != ''
                    GROUP BY {label_col}
                    HAVING COUNT(DISTINCT {EMBED_TEXT_COL}) > 1 AND {vsn_check}
                )
                SELECT d.*
                FROM {dedupe_table} d
                JOIN eligible e ON d.{label_col} = e.{label_col}
                ORDER BY RAND(42)
                LIMIT {n}
            """

        cursor = conn.cursor()
        cursor.execute(sample_query, params) if params else cursor.execute(sample_query)

        # Get column names from cursor description
        col_names = [desc[0] for desc in cursor.description]

        # source_row_count is already materialized on the dedupe table (how many
        # full-table rows collapsed into this one), so no per-row COUNT(*) needed.
        samples = [_row_to_dict(col_names, row) for row in cursor.fetchall()]

    except Exception as e:
        logger.error(f"Failed to fetch samples: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch samples: {e!s}")

    return {"samples": samples}


class SearchRequest(BaseModel):
    """Search request parameters."""

    query: str
    k: int = 10
    query_type: str = "HYBRID"  # HYBRID or ANN
    filters: dict[str, str] = {}
    variant: str = "dedupe"  # dedupe or full
    # Cluster id of the record the query came from, when the query was seeded by
    # clicking a sample. Lets each result be marked same-cluster or not, which is
    # the whole point of eyeballing results against the customer's ground truth.
    query_cluster_id: str | None = None


@app.post("/api/search")
async def search(request: SearchRequest):
    """Search the index and return results with cluster match highlighting."""
    cfg = get_config()
    client = get_client()

    # Select index based on variant
    if request.variant == "full":
        index_name = f"{_catalog}.{_schema}.po_products_ai_search_index"
    else:
        index_name = f"{_catalog}.{_schema}.po_products_ai_search_dedupe_index"

    # Get the actual synced columns from the index metadata. The GET response does
    # NOT echo back `columns_to_sync` (that field only exists on the create
    # request), so it is virtually always absent -- an empty result here is the
    # norm, not an error, and must fall back to the config just like an exception
    # does. Without the fallback the query goes out with no `columns` and the
    # endpoint rejects it with "Field 'columns' must be specified".
    synced_cols = []
    try:
        idx_response = client.api_client.do("GET", f"/api/2.0/vector-search/indexes/{index_name}")
        synced_cols = idx_response.get("delta_sync_index_spec", {}).get("columns_to_sync") or []
    except Exception as e:
        logger.warning(f"Failed to read index metadata for {index_name}: {e}")
    if not synced_cols:
        # Fallback to config-based columns.
        synced_cols = columns_to_sync(cfg)

    # Query the index
    try:
        # Goes through the shared REST helper: the SDK's query_index() can only
        # send filters_json, which this storage-optimized endpoint rejects, so it
        # cannot filter at all. col_names comes off the response manifest, which
        # carries the trailing `score` column the request did not ask for.
        col_names, data = query_index(
            client,
            index_name=index_name,
            columns=synced_cols,
            query_text=request.query,
            num_results=request.k + OVERFETCH,
            query_type=request.query_type,
            filters=request.filters,
        )

        label_col = ai_search_cfg(cfg)["label_col"]
        results = []
        for row in data[: request.k]:  # Return only top k (not k+OVERFETCH)
            result_dict = {name: _jsonable(val) for name, val in zip(col_names, row)}
            if request.query_cluster_id is not None:
                result_dict["cluster_match"] = str(result_dict.get(label_col)) == str(request.query_cluster_id)
            results.append(result_dict)

        return {"results": results, "query_cluster_id": request.query_cluster_id}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        # Provide more helpful error message
        error_msg = str(e)
        if "Something went wrong unexpectedly" in error_msg:
            error_msg = (
                "Vector Search index is not ready yet. It may still be syncing. Please wait a moment and try again."
            )
        raise HTTPException(status_code=503, detail=error_msg)


class EvaluateRequest(BaseModel):
    """Evaluate request parameters."""

    variant: str = "dedupe"
    multi_vsn_only: bool = True
    # 3 by default: the median cluster holds 2 distinct products, so a larger k
    # asks for more matches than the ground truth contains.
    k: int = 3
    n: int = 200
    query_type: str = "HYBRID"


@app.post("/api/evaluate")
async def evaluate(request: EvaluateRequest):
    """Run recall evaluation and return stats."""
    cfg = get_config()
    client = get_client()
    flat = ai_search_cfg(cfg)

    label_col = flat["label_col"]
    id_col = flat["id_col"]

    if request.variant == "full":
        table = f"{_catalog}.{_schema}.po_products_ai_search"
        index_name = f"{_catalog}.{_schema}.po_products_ai_search_index"
    else:
        table = f"{_catalog}.{_schema}.po_products_ai_search_dedupe"
        index_name = f"{_catalog}.{_schema}.po_products_ai_search_dedupe_index"

    try:
        # Sample rows using Spark
        samples = _sample_with_siblings(
            table,
            label_col,
            id_col,
            request.n,
            multi_vsn_only=request.multi_vsn_only,
        )

        if not samples:
            raise ValueError(f"no eligible rows in {table} (multi_vsn_only={request.multi_vsn_only})")

        # Score via the shared batch scorer rather than a local copy of the loop.
        # The app and the CLI each had their own and reported different numbers for
        # the same index. score_batch also runs the index queries concurrently --
        # serially they are ~2.4s each, so n=200 took roughly eight minutes.
        stats = score_batch(
            client,
            index_name=index_name,
            samples=samples,
            id_col=id_col,
            label_col=label_col,
            k=request.k,
            query_type=request.query_type,
            siblings_key=SIBLINGS_KEY,
        )
        stats["population"] = "multi_vsn_clusters" if request.multi_vsn_only else "multi_text_clusters"
        stats["variant"] = request.variant
        logger.info(
            "precision@%s = %s | recall@%s = %s (%s)",
            request.k,
            stats["precision_at_k"],
            request.k,
            stats["recall_at_k"],
            stats["population"],
        )
        return stats
    except Exception as e:
        logger.error(f"Evaluate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {e!s}")


def _sample_with_siblings(
    table: str,
    label_col: str,
    id_col: str,
    n: int,
    seed: int = 42,
    multi_vsn_only: bool = False,
) -> list[dict[str, Any]]:
    """Sample rows using SQL via databricks-sql-connector."""
    try:
        conn = get_sql_connection()

        # Step 1: Find eligible clusters (>1 distinct embed_text, optionally >1 distinct VSN)
        vsn_check = (
            "COUNT(DISTINCT CONCAT_WS('|', VENDOR_NAME, VENDOR_STYLE)) > 1"
            if multi_vsn_only
            else "1=1"
        )
        # Eligibility stays server-side: a cluster qualifies only if it holds more
        # than one DISTINCT embed_text, so there is a sibling to find that is not
        # a verbatim copy of the query. Materializing the ids client-side and
        # inlining them would build a query megabytes wide.
        sample_query = f"""
            WITH eligible AS (
                SELECT {label_col}, COUNT(DISTINCT {EMBED_TEXT_COL}) AS {SIBLINGS_KEY}
                FROM {table}
                WHERE TRIM({label_col}) != ''
                GROUP BY {label_col}
                HAVING COUNT(DISTINCT {EMBED_TEXT_COL}) > 1 AND {vsn_check}
            )
            SELECT t.*, e.{SIBLINGS_KEY}
            FROM {table} t
            JOIN eligible e ON t.{label_col} = e.{label_col}
            ORDER BY RAND({seed})
            LIMIT {n}
        """

        cursor = conn.cursor()
        cursor.execute(sample_query)

        # Get column names from cursor description
        col_names = [desc[0] for desc in cursor.description]

        samples = [_row_to_dict(col_names, row) for row in cursor.fetchall()]
        if not samples:
            raise ValueError(
                f"no eligible rows in {table} (multi_vsn_only={multi_vsn_only}); cannot measure recall"
            )
        return samples
    except Exception as e:
        logger.error(f"Failed to sample with siblings: {e}")
        raise


@app.get("/api/data")
async def get_data(
    table: str = Query("po_products_ai_search_dedupe", description="Table to browse"),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return paginated data from a table with server-side pagination.

    Supports browsing po_products_ai_search, po_products_ai_search_dedupe,
    po_products_ai_search_bronze, and po_products_ai_search_quarantine tables.
    Uses LIMIT and OFFSET for server-side pagination.
    """
    # Allowlist of safe table names
    allowed_tables = {
        "po_products_ai_search",
        "po_products_ai_search_dedupe",
        "po_products_ai_search_bronze",
        "po_products_ai_search_quarantine",
    }

    if table not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table: {table}")

    try:
        conn = get_sql_connection()
        full_table = f"{_catalog}.{_schema}.{table}"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM {full_table}"
        cursor = conn.cursor()
        cursor.execute(count_query)
        total = cursor.fetchone()[0]

        # Fetch paginated rows
        data_query = f"SELECT * FROM {full_table} LIMIT {limit} OFFSET {offset}"
        cursor = conn.cursor()
        cursor.execute(data_query)

        # Get column names
        col_names = [desc[0] for desc in cursor.description]

        # Convert rows to dicts
        rows = [_row_to_dict(col_names, row) for row in cursor.fetchall()]

        return {
            "rows": rows,
            "columns": col_names,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e!s}")


# Serve static HTML separately
@app.get("/{file_path:path}", include_in_schema=False)
async def serve_static(file_path: str):
    """Serve static files."""
    if not file_path or file_path == "/":
        file_path = "index.html"

    file_full_path = os.path.join(os.path.dirname(__file__), file_path)

    # Security: prevent path traversal
    if not os.path.abspath(file_full_path).startswith(os.path.abspath(os.path.dirname(__file__))):
        raise HTTPException(status_code=403, detail="Access denied")

    if os.path.isfile(file_full_path):
        with open(file_full_path, "r") as f:
            content = f.read()
        if file_path.endswith(".html"):
            from fastapi.responses import HTMLResponse

            return HTMLResponse(content)
        elif file_path.endswith(".css"):
            from fastapi.responses import Response

            return Response(content, media_type="text/css")
        elif file_path.endswith(".js"):
            from fastapi.responses import Response

            return Response(content, media_type="application/javascript")

    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn

    # Set default profile if not set
    if "DATABRICKS_CONFIG_PROFILE" not in os.environ:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = "AZURE-SA-WORKSPACE"

    uvicorn.run(app, host="127.0.0.1", port=8000)
