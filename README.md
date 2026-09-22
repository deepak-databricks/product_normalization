# product_normalization

Product identity resolution for TJX purchase-order data: group rows that refer to the
same physical product.

* `src`: Python source code for this project.
* `resources`:  Resource configurations (jobs, pipelines, etc.)
* `tests`: Unit tests for the shared Python code.
* `app`: local FastAPI page for driving the index by hand — see below.
* `assets/data`: Customer PO extract and the small `po_data_sample.csv` used by the
  tests. **Gitignored** — the data is customer-confidential and is moved via the UC
  volume, not the repo. Tests that need the sample skip cleanly when it is absent.
* `config.yaml`: which columns drive matching. **Single source of truth** — the job reads
  it, so changing `ai_search.embedding_cols` changes what gets embedded with no code edit.

## The `create_ai_search_index` job

Three tasks, chained. Each is independently re-runnable so a failure late in the chain
doesn't force a re-read of the CSV or a re-embed of ~2M rows.

| task | does |
|---|---|
| `ingest_raw` | UC volume CSV → `<table>_bronze`, verbatim. Asserts 1,999,911 rows. |
| `build_search_table` | Normalizes null sentinels, gates `EXPORT_COUNTRY`, derives `embed_text`, quarantines broken rows, enables Change Data Feed. |
| `create_index` | Creates the Vector Search index (or syncs it if it exists). |

Tables created in `${catalog}.${schema}`:

| table | contents |
|---|---|
| `po_products_ai_search` | the indexable table |
| `po_products_ai_search_bronze` | raw CSV as loaded |
| `po_products_ai_search_quarantine` | the ~3,176 field-shifted rows |

### `embed_text`

Vector Search's managed embeddings read a **STRING** column, never a STRUCT, so the
columns in `ai_search.embedding_cols` are flattened into one string. No struct column is
materialized — those fields already exist as columns of their own.

Each value is prefixed with its lowercased column name, which gives the model a field cue
(`vendor_style: 710849298007` reads differently than a bare digit string). The prefix is
derived from the column name, so `embedding_cols` alone controls the output:

```
vendor_name: RALPH LAUREN EUR/BLUE LABEL | vendor_style: 710849298007 |
product_description: 601 POPLIN SPORT SHIRT | department_description: BEST BRANDS | ...
```

NULL fields drop out entirely rather than leaving an empty `export_country:` fragment.

### Defects handled (all present in the data as the customer sent it)

1. **Field shift** — 3,176 rows (0.159%). An inch mark in a product name (`12" CERAMIC
   WOK`) was double-escaped at export, breaking CSV parsing and pushing later values one
   column right. Detected via `CONFIDENCE_SCORE` not being numeric. These rows carry
   garbage in exactly the fields being embedded, so they are quarantined rather than
   indexed. Override with `--on-malformed {quarantine,drop,keep}`.
2. **Embedded newlines** — 14 of the 25 source parts have newlines inside quoted fields,
   and none ends with a trailing newline. The reader therefore *requires* `multiLine=true`
   and `escape='"'`; the row-count assertion in `ingest` is the tripwire if either is lost.
3. **Null sentinels** — literal `None`/`NA`/`-` strings mean "missing" but aren't empty.
   Normalized to NULL before embedding, matching whole values only so a vendor named
   `ST. JOHN` survives.
4. **Structural nullity** — `EXPORT_COUNTRY` is 64% missing, but 0% on IMPORT rows vs
   59.8% on DOMESTIC. Left raw it leaks the import/domestic flag into similarity, so it is
   NULLed on non-IMPORT rows.

### Running it

```bash
# One-time: upload the CSV to the volume
databricks fs cp assets/data/po_data_all.csv \
  dbfs:/Volumes/users/akil_thomas/raw_data/po/po_data_all.csv --overwrite

databricks bundle deploy -t dev
databricks bundle run create_ai_search_index -t dev
```

The index sync of ~2M rows continues server-side after the job returns; poll with
`databricks vector-search-indexes get-index <catalog>.<schema>.po_products_ai_search_index`.

### Checking whether it actually works

Every task above can succeed while retrieval is useless, so measure it. Once the index
reports ready:

```bash
python -m product_normalization evaluate --catalog users --schema akil_thomas --k 3
```

This reports **recall@k against TJX's own cluster IDs**: for each sampled row, query the
index with that row's `embed_text` and check whether a *different* row carrying the same
`PS_PRODUCT_ID` comes back in the top k. Only rows whose cluster has a sibling are
sampled — a singleton cluster has no correct answer to find.

`self_not_returned` in the output should be near zero; if a row can't retrieve *itself*,
the index is stale or the sync is incomplete.

### Metric: precision@k, not recall

The headline number is **precision@k** — of the products returned, how many share
the input's `PS_PRODUCT_ID`. Recall (did *any* sibling come back) is reported
alongside, but it answers a narrower question than a practitioner scanning a
result list actually cares about.

The precision denominator is `min(k, siblings_available)`, **not** k. This matters
more than the metric choice: 49.9% of clusters hold exactly one distinct product
and the median is 2, so dividing by a fixed k=10 would score a system that
returned every sibling that exists as 0.2. Only 12.8% of clusters even contain
enough products to fill 10 slots. `slots_capped_by_ground_truth` reports how often
that ceiling binds.

`k` defaults to **3** for the same reason.

| variant | precision@3 | recall@3 |
|---|---|---|
| `dedupe` | **0.884** | 0.953 |
| `full` | 0.730 | 0.700 |

Measured on multi-VSN clusters with HYBRID, n=150. Dedupe wins because duplicate
`embed_text` rows crowd the result list — one query against `full` returned five
byte-identical rows in its top five, wasting four of the five slots.

#### Building the `dedupe` variant

The `create_ai_search_index` job builds only the **`full`** table and index. The
`dedupe` variant — the one that scores highest above — is a separate step you run
after the job, collapsing the table to one row per distinct `embed_text` and
indexing that sibling:

```bash
# 1. Collapse the full table to one row per distinct embed_text
python -m product_normalization build-dedupe --catalog users --schema <schema>

# 2. Build the dedupe index (writes po_products_ai_search_dedupe_index)
python -m product_normalization create-index --variant dedupe \
  --catalog users --schema <schema> --endpoint cfc_ai_search

# 3. Score it — this reproduces the precision@3 = 0.884 row above
python -m product_normalization evaluate --variant dedupe --multi-vsn-only \
  --query-type HYBRID --catalog users --schema <schema>
```

`--variant dedupe` points every subcommand at the `_dedupe` table and
`_dedupe_index` so both approaches can be built and measured side by side. The
local app exposes the same toggle (`full` / `dedupe`) once both indexes exist.

### The local app

```bash
uv sync --extra app          # fastapi, uvicorn, databricks-sql-connector, python-dotenv
cp .env.example .env         # then fill it in — see below
PYTHONPATH=src python3 app/main.py     # http://127.0.0.1:8000
```

`app/main.py` auto-loads `.env` (via python-dotenv) at startup, so put your
settings there rather than exporting them by hand. The four variables the app
actually reads — all with personal hardcoded defaults you should override:

| var | what it controls |
|---|---|
| `DATABRICKS_CONFIG_PROFILE` | CLI profile the WorkspaceClient loads (default `AZURE-SA-WORKSPACE`) |
| `DATABRICKS_CATALOG` | catalog holding the tables (default `users`) |
| `DATABRICKS_SCHEMA` | schema holding the tables — **set this to your own**, or the app queries someone else's data |
| `DATABRICKS_WAREHOUSE_HTTP_PATH` | SQL warehouse for sampling/filter values (Connection details in the workspace) |

Log in first with `databricks auth login --profile AZURE-SA-WORKSPACE`.

Search box, ANN/HYBRID toggle, tunable `k`, filter dropdowns, sample records that
populate the query on click, and an Evaluate button that scores precision/recall
over a sample. Sampling and filter values go through a SQL warehouse rather than
Spark — Databricks Connect takes ~10s to start a session, which makes every click
feel broken.

### Filtering: use `filter_string`

`cfc_ai_search` is a **STORAGE_OPTIMIZED** endpoint. Those take a SQL predicate in
the REST field `filter_string`; the dict form (`filters_json`) is rejected. Two
traps:

1. `WorkspaceClient.vector_search_indexes.query_index()` only exposes
   `filters_json`, so **the SDK method cannot filter on this endpoint at all**.
   Use `evaluate.query_index()`, which calls REST directly.
2. The query endpoint **silently ignores unrecognised body keys**. Sending
   `filters` (the name the docs and Python client use) returns HTTP 200 with
   unfiltered results. Assert that a filtered query narrows the result set; never
   infer success from the absence of an error.

If precision disappoints, the levers in order of cost:
1. **Hybrid search** — the index is created with `index_subtype: HYBRID`, so BM25 keyword
   scoring is already available with no rebuild. Pass `query_type="HYBRID"` to bring the
   VSN in as an exact-match term, which pure semantic search can miss on digit strings
   like `710849298007`.
2. Drop the weakest secondary fields (`BUYER_NAME`, `EXPORT_COUNTRY`) from
   `ai_search.embedding_cols` — no code change needed.
3. Self-managed embeddings with per-tier weighting from `ai_search.field_tiers`.


## Getting started

Choose how you want to work on this project:

(a) Directly in your Databricks workspace, see
    https://docs.databricks.com/dev-tools/bundles/workspace.

(b) Locally with an IDE like Cursor or VS Code, see
    https://docs.databricks.com/dev-tools/vscode-ext.html.

(c) With command line tools, see https://docs.databricks.com/dev-tools/cli/databricks-cli.html

If you're developing with an IDE, dependencies for this project should be installed using uv:

*  Make sure you have the UV package manager installed.
   It's an alternative to tools like pip: https://docs.astral.sh/uv/getting-started/installation/.
*  Run `uv sync --dev` to install the project's dependencies.


# Using this project using the CLI

The Databricks workspace and IDE extensions provide a graphical interface for working
with this project. It's also possible to interact with it directly using the CLI:

1. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks configure
    ```

2. To deploy a development copy of this project, type:
    ```
    $ databricks bundle deploy --target dev
    ```
    (Note that "dev" is the default target, so the `--target` parameter
    is optional here.)

    This deploys everything that's defined for this project.

3. Similarly, to deploy a production copy, type:
   ```
   $ databricks bundle deploy --target prod
   ```

4. To run a job or pipeline, use the "run" command:
   ```
   $ databricks bundle run
   ```

5. Finally, to run tests locally, use `pytest`:
   ```
   $ uv run pytest
   ```
