"""Derive ``embed_text`` -- the one column Vector Search embeds.

Vector Search's managed embeddings read a STRING column and never a STRUCT, so
the fields named in ``ai_search.embedding_cols`` are flattened into a single
string here. No struct column is materialized: those fields already exist as
columns of their own, so a struct would only duplicate them.

Each value is prefixed with its column name, lowercased, which gives the model a
field cue -- ``vendor_style: 710849298007`` reads differently than a bare digit
string. The prefix is derived from the column name rather than configured
separately, so ``embedding_cols`` alone determines what gets embedded.
"""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

EMBED_TEXT_COL = "embed_text"

# Separator between field fragments. Rare enough in the data that it will not be
# confused for content: the PO extract uses commas and slashes, not pipes.
FIELD_SEPARATOR = " | "


# Marker appended to a temporary copy of a column that was transformed purely for
# embedding (see build_table.GATED_SUFFIX). It must not surface in the text.
_INTERNAL_SUFFIX = "_gated_for_embedding"


def field_prefix(col_name: str) -> str:
    """Prefix shown before a field's value (``VENDOR_STYLE`` -> ``vendor_style``).

    Strips the internal marker so a column gated only for embedding still reads as
    its real name -- ``export_country:``, never
    ``export_country_gated_for_embedding:``.
    """
    return col_name.removesuffix(_INTERNAL_SUFFIX).lower()


def build_embed_text(embedding_cols: list[str], separator: str = FIELD_SEPARATOR) -> Column:
    """Assemble ``embed_text`` from the given columns, in order.

    NULL fields drop out entirely rather than leaving an empty fragment such as
    ``export_country:``, because ``concat_ws`` skips NULL arguments. Callers are
    expected to have normalized sentinels to NULL first (see
    :func:`~product_normalization.ai_search.cleaning.normalize_sentinels`), so a
    literal ``"None"`` never reaches the embedding.
    """
    if not embedding_cols:
        raise ValueError("embedding_cols is empty; nothing to embed")

    parts: list[Column] = []
    for col_name in embedding_cols:
        value = F.col(col_name)
        # Keep the fragment NULL (not "prefix: ") when the value is NULL, so
        # concat_ws drops the whole thing.
        parts.append(F.when(value.isNotNull(), F.concat(F.lit(f"{field_prefix(col_name)}: "), value)))

    return F.concat_ws(separator, *parts)
