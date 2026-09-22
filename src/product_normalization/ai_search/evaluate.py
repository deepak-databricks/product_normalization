"""Measure whether the index actually retrieves same-product rows.

Everything upstream of this is plumbing: it can all succeed while retrieval is
useless. This is the check that says whether the embedding choice works.

Two metrics, both scored against the customer's own ``PS_PRODUCT_ID`` clusters
from the same index query so they can never disagree:

``recall@k``
    Did *any* same-cluster sibling come back in the top k? Answers "can we find a
    match at all", and is the number the earlier full-vs-dedupe comparison used.

``precision@k`` (the headline)
    What *fraction* of the k returned products share the query's cluster? This is
    the practitioner-facing question -- someone looking at a result list wants
    most of it to be right, not just one row buried in it.

    The denominator is ``min(k, siblings_available)``, not k. Half of all clusters
    hold exactly one distinct product and the median is 2, so a fixed-k
    denominator would score a perfect system as failing: return both existing
    siblings at k=10 and naive precision reads 0.2 while the system did
    everything possible. Normalising by what could have been retrieved keeps the
    metric honest about the ground truth's shape. ``slots_available`` and
    ``slots_capped_by_ground_truth`` are reported so the ceiling stays visible.

Only rows whose cluster has at least one *textually different* sibling are
eligible -- a singleton cluster has no correct answer, so scoring it would just
depress both numbers for no reason.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config import ai_search_cfg
from .embed_text import EMBED_TEXT_COL

logger = logging.getLogger(__name__)


VSN_COLS = ("VENDOR_NAME", "VENDOR_STYLE")

# Extra results fetched beyond k so exact-duplicate texts can be discarded without
# starving the top-k window. The worst-duplicated text repeats 9,386 times, so this
# cannot guarantee a clean k for every row -- but it covers the common case.
OVERFETCH = 50


def build_filter_string(filters: dict[str, Any]) -> str | None:
    """Render {col: value} as the SQL predicate storage-optimized endpoints want.

    STORAGE_OPTIMIZED endpoints reject the dict form (``filters_json``) and take a
    SQL string instead. Values are single-quoted with embedded quotes doubled;
    lists become ``IN (...)``.
    """
    if not filters:
        return None

    def literal(v: Any) -> str:
        return "'" + str(v).replace("'", "''") + "'"

    clauses = []
    for col, value in filters.items():
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            values = [v for v in value if v is not None and v != ""]
            if not values:
                continue
            clauses.append(f"{col} IN ({', '.join(literal(v) for v in values)})")
        else:
            clauses.append(f"{col} = {literal(value)}")
    return " AND ".join(clauses) if clauses else None


def query_index(
    client: WorkspaceClient,
    index_name: str,
    columns: list[str],
    query_text: str,
    num_results: int,
    query_type: str | None = None,
    filters: dict[str, Any] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Query an index, via REST so that filtering actually works.

    ``WorkspaceClient.vector_search_indexes.query_index()`` only exposes
    ``filters_json``, which STORAGE_OPTIMIZED endpoints reject outright -- so the
    SDK method cannot filter on this endpoint at all. The REST body field is
    ``filter_string``.

    Worse, the query endpoint silently ignores body keys it does not recognise, so
    sending ``filters`` (the name the docs and Python client use) returns
    unfiltered results with no error. Anything that looks like a working filtered
    query has to be asserted, not assumed.

    Returns ``(column_names, rows)``. Column names come off the response manifest
    because a trailing ``score`` column is appended to what was requested.
    """
    body: dict[str, Any] = {
        "columns": columns,
        "query_text": query_text,
        "num_results": num_results,
    }
    if query_type:
        body["query_type"] = query_type
    filter_string = build_filter_string(filters or {})
    if filter_string:
        body["filter_string"] = filter_string

    response = client.api_client.do("POST", f"/api/2.0/vector-search/indexes/{index_name}/query", body=body)
    result = response.get("result") or {}
    manifest = response.get("manifest") or {}
    col_names = [c["name"] for c in manifest.get("columns", [])] or [*columns, "score"]
    # A filter that matches nothing omits data_array entirely rather than
    # returning an empty list.
    return col_names, result.get("data_array") or []


def score_one(
    client: WorkspaceClient,
    index_name: str,
    query_text: str,
    query_id: str,
    query_label: str,
    id_col: str,
    label_col: str,
    k: int,
    siblings_available: int | None = None,
    query_type: str | None = None,
) -> dict[str, Any]:
    """Query the index with one row and score the result list.

    Split out so the CLI and the web app score identically off a single
    implementation -- they differ only in how they *sample*, and an earlier
    divergence here produced two incompatible numbers for the same index.

    ``siblings_available`` is how many textually-distinct same-cluster products
    exist to be found. Pass it to get a precision denominator of
    ``min(k, siblings_available)``; omit it and the denominator is the number of
    distinct results actually returned.
    """
    _, data = query_index(
        client,
        index_name=index_name,
        columns=[id_col, label_col, EMBED_TEXT_COL],
        query_text=query_text,
        # Over-fetch: exact-duplicate texts crowd the head of the result list and
        # would otherwise fill the whole window before a genuinely different
        # sibling appears.
        num_results=k + OVERFETCH,
        # HYBRID adds BM25 keyword scoring, which helps on VSN digit strings that
        # pure vector similarity treats as near-meaningless tokens.
        query_type=query_type,
    )
    returned = [(str(r[0]), str(r[1]), str(r[2])) for r in data]

    self_returned = any(rid == str(query_id) for rid, _, _ in returned)

    # Drop exact-text copies of the query, then keep the top k of what remains. A
    # hit must be a product whose text actually differs -- otherwise we are just
    # rediscovering copies of the query.
    qt = str(query_text)
    distinct = [(rid, lbl) for rid, lbl, txt in returned if txt != qt][:k]
    matched = sum(1 for _, lbl in distinct if lbl == str(query_label))

    if siblings_available is not None:
        denominator = min(k, siblings_available)
    else:
        denominator = len(distinct)

    return {
        "returned": len(distinct),
        "matched": matched,
        "slots_available": denominator,
        "precision": (matched / denominator) if denominator else 0.0,
        "hit": matched > 0,
        "self_returned": self_returned,
    }


def sample_with_siblings(
    spark: SparkSession,
    table: str,
    id_col: str,
    label_col: str,
    n: int,
    seed: int = 42,
    multi_vsn_only: bool = False,
):
    """Rows whose cluster contains at least one other row.

    With ``multi_vsn_only``, restrict to clusters that span more than one VSN.
    That matters because 61.6% of multi-row clusters share a single VSN and are
    already solved by exact match -- including them flatters recall without
    telling you anything about the embedding. The remaining 38.4% are the
    population semantic search actually has to earn.
    """
    df = spark.table(table).filter(F.trim(F.col(label_col)) != "")

    # Count DISTINCT embed_text per cluster, not rows. 79.6% of rows share their
    # embed_text with another row (one text repeats 9,386 times), because the same
    # product recurs across POs and months with identical descriptive fields. A row
    # whose exact text appears thousands of times retrieves a "sibling" trivially,
    # which measures duplication rather than semantic matching and pins recall at
    # 1.0. Requiring a cluster to hold more than one distinct text means the match
    # has to bridge an actual textual difference.
    grouped = df.groupBy(label_col).agg(
        F.countDistinct(EMBED_TEXT_COL).alias("_ntext"),
        F.countDistinct(F.concat_ws("|", *[F.col(c) for c in VSN_COLS])).alias("_nvsn"),
    )
    eligible_clusters = grouped.filter(F.col("_ntext") > 1)
    if multi_vsn_only:
        eligible_clusters = eligible_clusters.filter(F.col("_nvsn") > 1)

    # Carry _ntext through: precision needs to know how many textually-distinct
    # products the cluster holds, since that caps what any system could return.
    eligible = df.join(eligible_clusters.select(label_col, "_ntext"), on=label_col, how="inner")
    total = eligible.count()
    if total == 0:
        raise ValueError(f"no eligible rows in {table} (multi_vsn_only={multi_vsn_only}); cannot measure recall")

    fraction = min(1.0, (n * 3.0) / total)
    return eligible.sample(withReplacement=False, fraction=fraction, seed=seed).limit(n).collect()


SIBLINGS_COL = "_ntext"

# Index queries are network-bound at ~2.4s each, so scoring 200 samples serially
# takes ~8 minutes. They are independent and the index is read-only, so run them
# in parallel. Kept modest to avoid hammering the serving endpoint.
SCORE_WORKERS = 12


def score_batch(
    client: WorkspaceClient,
    index_name: str,
    samples: list[dict[str, Any]],
    id_col: str,
    label_col: str,
    k: int,
    query_type: str | None = None,
    siblings_key: str = SIBLINGS_COL,
    workers: int = SCORE_WORKERS,
) -> dict[str, Any]:
    """Score every sample and aggregate into the stats dict.

    Shared by the CLI and the web app so a single definition produces both
    numbers. Queries run concurrently because each one is an independent
    read-only network call.
    """

    def one(sample: dict[str, Any]) -> dict[str, Any]:
        # A cluster holding n distinct texts offers n-1 findable siblings, since
        # the query's own text is excluded from the results.
        available = max(int(sample.get(siblings_key) or 1) - 1, 0)
        scored = score_one(
            client,
            index_name=index_name,
            query_text=sample[EMBED_TEXT_COL],
            query_id=sample[id_col],
            query_label=sample[label_col],
            id_col=id_col,
            label_col=label_col,
            k=k,
            siblings_available=available,
            query_type=query_type,
        )
        scored["_available"] = available
        return scored

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, samples))

    sampled = len(results)
    hits = sum(1 for r in results if r["hit"])
    self_missing = sum(1 for r in results if not r["self_returned"])
    precision_sum = sum(r["precision"] for r in results)
    matched_total = sum(r["matched"] for r in results)
    slots_total = sum(r["slots_available"] for r in results)
    capped = sum(1 for r in results if r["_available"] < k)

    return {
        "sampled": sampled,
        "k": k,
        # Mean over queries of matched / min(k, siblings_available).
        "precision_at_k": round(precision_sum / sampled, 4) if sampled else 0.0,
        # Pooled rather than averaged per query, so large clusters can move it.
        # The two diverging signals that cluster size is driving the result.
        "micro_precision_at_k": round(matched_total / slots_total, 4) if slots_total else 0.0,
        "recall_at_k": round(hits / sampled, 4) if sampled else 0.0,
        "hits": hits,
        "matched_products": matched_total,
        "slots_available": slots_total,
        # How many queries had fewer than k findable siblings. High values mean the
        # ground truth, not retrieval, is bounding the score.
        "slots_capped_by_ground_truth": capped,
        "self_not_returned": self_missing,
        "query_type": query_type or "ANN",
        "note": "precision denominator is min(k, distinct same-cluster products excluding the query)",
    }


def recall_at_k(
    spark: SparkSession,
    client: WorkspaceClient,
    index_name: str,
    table: str,
    cfg: dict[str, Any],
    n: int = 200,
    k: int = 10,
    multi_vsn_only: bool = False,
    query_type: str | None = None,
) -> dict[str, Any]:
    """Precision and recall at k over sampled rows, scored via ``score_one``."""
    flat = ai_search_cfg(cfg)
    id_col, label_col = flat["id_col"], flat["label_col"]

    rows = sample_with_siblings(spark, table, id_col, label_col, n, multi_vsn_only=multi_vsn_only)
    logger.info(
        "evaluating %s sampled rows at k=%s (multi_vsn_only=%s, query_type=%s)",
        len(rows),
        k,
        multi_vsn_only,
        query_type or "ANN",
    )

    stats = score_batch(
        client,
        index_name=index_name,
        samples=[row.asDict() for row in rows],
        id_col=id_col,
        label_col=label_col,
        k=k,
        query_type=query_type,
    )
    stats = {
        **stats,
        "population": "multi_vsn_clusters" if multi_vsn_only else "multi_text_clusters",
        "query_type": query_type or "ANN",
        "note": "precision denominator is min(k, distinct same-cluster products excluding the query)",
    }
    logger.info(
        "precision@%s = %s | recall@%s = %s (%s)",
        k,
        stats["precision_at_k"],
        k,
        stats["recall_at_k"],
        stats["population"],
    )
    return stats
