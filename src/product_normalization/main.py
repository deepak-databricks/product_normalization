"""Entry point for the create_ai_search_index job.

Three subcommands, one per job task, so a failure at 2M-row scale can be retried
without repeating the upload or the embedding:

    ingest        volume CSV  -> bronze Delta table
    build-table   bronze      -> indexable table (+ quarantine)
    create-index  table       -> Vector Search index
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

logger = logging.getLogger("product_normalization")


def _spark():
    """Resolve a SparkSession, working both on a cluster and locally.

    On a Databricks cluster the plain builder returns the ambient session. Run
    locally, that same call raises because only remote sessions are supported, so
    fall back to Databricks Connect -- which is how ``evaluate`` gets used from a
    laptop against an already-built table.
    """
    from pyspark.sql import SparkSession

    try:
        return SparkSession.builder.getOrCreate()
    except RuntimeError:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default="po_products_ai_search", help="base table name")
    parser.add_argument("--config-path", default=None, help="override the packaged config.yaml")


def _add_variant(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--variant",
        choices=("full", "dedupe"),
        default="full",
        help="which table/index to act on: 'full' (every row) or 'dedupe' (one row per distinct embed_text)",
    )


def _names(args: argparse.Namespace) -> dict[str, str]:
    """Table and index names.

    ``--variant dedupe`` points target/index at the deduplicated sibling so both
    approaches can be built and measured side by side.
    """
    prefix = f"{args.catalog}.{args.schema}"
    base = args.table
    dedupe_table = f"{prefix}.{base}_dedupe"
    names = {
        "bronze": f"{prefix}.{base}_bronze",
        "full": f"{prefix}.{base}",
        "dedupe": dedupe_table,
        "quarantine": f"{prefix}.{base}_quarantine",
        "full_index": f"{prefix}.{base}_index",
        "dedupe_index": f"{prefix}.{base}_dedupe_index",
    }
    variant = getattr(args, "variant", "full")
    names["target"] = names["dedupe"] if variant == "dedupe" else names["full"]
    names["index"] = names["dedupe_index"] if variant == "dedupe" else names["full_index"]
    return names


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ai_search.ingest import ingest

    names = _names(args)
    expected = None if args.no_row_check else args.expected_rows
    count = ingest(_spark(), args.source_path, names["bronze"], expected_rows=expected)
    print(f"ingested {count:,} rows into {names['bronze']}")
    return 0


def cmd_build_table(args: argparse.Namespace) -> int:
    from .ai_search.build_table import build

    names = _names(args)
    cfg = load_config(args.config_path)
    stats = build(
        _spark(),
        bronze_table=names["bronze"],
        target_table=names["target"],
        quarantine_table=names["quarantine"],
        cfg=cfg,
        on_malformed=args.on_malformed,
    )
    print(f"built {names['target']}")
    for key, value in stats.items():
        print(f"  {key:18} {value:,}")
    return 0


def cmd_build_dedupe(args: argparse.Namespace) -> int:
    from .ai_search.dedupe import build_dedupe_table

    names = _names(args)
    cfg = load_config(args.config_path)
    stats = build_dedupe_table(_spark(), source_table=names["full"], target_table=names["dedupe"], cfg=cfg)
    print(f"built {names['dedupe']}")
    for key, value in stats.items():
        print(f"  {key:18} {value:,}" if isinstance(value, int) else f"  {key:18} {value}")
    return 0


def cmd_create_index(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from .ai_search.index import ensure_endpoint, ensure_index, wait_until_ready

    names = _names(args)
    cfg = load_config(args.config_path)
    client = WorkspaceClient()

    ensure_endpoint(client, args.endpoint)
    action = ensure_index(
        client,
        index_name=names["index"],
        endpoint=args.endpoint,
        source_table=names["target"],
        embedding_endpoint=args.embedding_endpoint,
        cfg=cfg,
    )
    print(f"{action} index {names['index']}")

    ready = wait_until_ready(client, names["index"], minutes=args.wait_minutes)
    if not ready and args.wait_minutes > 0:
        print(f"index {names['index']} is still syncing; this is expected for a first build")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from .ai_search.evaluate import recall_at_k

    names = _names(args)
    cfg = load_config(args.config_path)
    stats = recall_at_k(
        _spark(),
        WorkspaceClient(),
        index_name=names["index"],
        table=names["target"],
        cfg=cfg,
        n=args.sample_size,
        k=args.k,
        multi_vsn_only=args.multi_vsn_only,
        query_type=args.query_type,
    )
    print(f"retrieval quality for {names['index']}")
    for key, value in stats.items():
        print(f"  {key:28} {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="product_normalization")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="load the raw CSV into a bronze table")
    _add_common(p_ingest)
    p_ingest.add_argument("--source-path", required=True, help="UC volume path to the CSV")
    p_ingest.add_argument("--expected-rows", type=int, default=1_999_911)
    p_ingest.add_argument(
        "--no-row-check",
        action="store_true",
        help="skip the row-count assertion (use only with a different extract)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build-table", help="derive embed_text and write the indexable table")
    _add_common(p_build)
    p_build.add_argument(
        "--on-malformed",
        choices=("quarantine", "drop", "keep"),
        default="quarantine",
        help="what to do with field-shifted rows (default: quarantine)",
    )
    p_build.set_defaults(func=cmd_build_table)

    p_dd = sub.add_parser("build-dedupe", help="collapse the table to one row per distinct embed_text")
    _add_common(p_dd)
    p_dd.set_defaults(func=cmd_build_dedupe)

    p_index = sub.add_parser("create-index", help="create or sync the Vector Search index")
    _add_common(p_index)
    _add_variant(p_index)
    p_index.add_argument("--endpoint", required=True)
    p_index.add_argument("--embedding-endpoint", default="databricks-gte-large-en")
    p_index.add_argument(
        "--wait-minutes",
        type=float,
        default=0.0,
        help="poll for readiness; 0 returns immediately after triggering",
    )
    p_index.set_defaults(func=cmd_create_index)

    p_eval = sub.add_parser("evaluate", help="measure precision/recall@k against the customer's cluster IDs")
    _add_common(p_eval)
    _add_variant(p_eval)
    p_eval.add_argument("--sample-size", type=int, default=200)
    p_eval.add_argument(
        "--k",
        type=int,
        default=3,
        help="results per query. Default 3: the median cluster holds 2 distinct products, "
        "so a large k asks for more matches than the ground truth contains",
    )
    p_eval.add_argument(
        "--multi-vsn-only",
        action="store_true",
        help="sample only clusters spanning >1 VSN -- the cases exact match cannot already solve",
    )
    p_eval.add_argument(
        "--query-type",
        choices=("ANN", "HYBRID"),
        default=None,
        help="HYBRID adds BM25 keyword scoring (helps on VSN digit strings)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
