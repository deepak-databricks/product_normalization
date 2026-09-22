"""Precision@k scoring, in particular its denominator.

The denominator is the whole point of this metric. Half of all clusters hold
exactly one distinct product and the median is 2, so scoring against a fixed k
would mark a system that returned every existing sibling as failing. These tests
pin that behaviour, because a future "simplification" to `matched / k` would
silently make every number look bad.
"""

from types import SimpleNamespace

import pytest

from product_normalization.ai_search.evaluate import (
    OVERFETCH,
    build_filter_string,
    query_index,
    score_one,
)


class FakeApiClient:
    """Records the REST body it was asked to send and replays canned rows.

    Mirrors the real wire shape: query goes through client.api_client.do() because
    the SDK's query_index() cannot send filter_string.
    """

    def __init__(self, rows):
        self.rows = rows
        self.last_body = None
        self.last_path = None

    def do(self, method, path, body=None):
        self.last_path = path
        self.last_body = body
        return {
            "manifest": {"columns": [{"name": "REC_ID"}, {"name": "PS_PRODUCT_ID"}, {"name": "embed_text"}]},
            "result": {"data_array": self.rows},
        }


def client_returning(rows):
    return SimpleNamespace(api_client=FakeApiClient(rows))


def score(rows, k, siblings_available=None, query_text="q", query_label="C1"):
    client = client_returning(rows)
    return score_one(
        client,
        index_name="idx",
        query_text=query_text,
        query_id="r0",
        query_label=query_label,
        id_col="REC_ID",
        label_col="PS_PRODUCT_ID",
        k=k,
        siblings_available=siblings_available,
    )


def test_perfect_retrieval_of_a_two_product_cluster_scores_1_at_k3():
    """The case that motivated the metric.

    A cluster with 2 distinct products offers exactly 1 findable sibling. Return
    it and the system did everything possible -- that must read 1.0, not 1/3.
    """
    rows = [
        ("r0", "C1", "q"),  # the query itself, excluded by text
        ("r1", "C1", "sibling"),  # the one findable sibling
        ("r2", "C9", "unrelated"),
        ("r3", "C8", "unrelated too"),
    ]
    out = score(rows, k=3, siblings_available=1)
    assert out["matched"] == 1
    assert out["slots_available"] == 1
    assert out["precision"] == 1.0


def test_fixed_k_denominator_would_have_understated_that_case():
    """Guards the intent: matched/k gives 0.33 where the system was perfect."""
    rows = [("r0", "C1", "q"), ("r1", "C1", "sibling"), ("r2", "C9", "x")]
    out = score(rows, k=3, siblings_available=1)
    naive = out["matched"] / 3
    assert naive == pytest.approx(0.3333, abs=1e-4)
    assert out["precision"] == 1.0


def test_denominator_is_capped_by_k_when_many_siblings_exist():
    """A cluster with 20 siblings cannot earn credit for more than k slots."""
    rows = [("r0", "C1", "q")] + [(f"r{i}", "C1", f"s{i}") for i in range(1, 4)]
    out = score(rows, k=3, siblings_available=20)
    assert out["slots_available"] == 3
    assert out["matched"] == 3
    assert out["precision"] == 1.0


def test_half_right_reads_half():
    rows = [
        ("r0", "C1", "q"),
        ("r1", "C1", "good"),
        ("r2", "C7", "bad"),
        ("r3", "C1", "good two"),
        ("r4", "C8", "bad two"),
    ]
    out = score(rows, k=4, siblings_available=4)
    assert out["matched"] == 2
    assert out["slots_available"] == 4
    assert out["precision"] == 0.5


def test_query_text_duplicates_never_count_as_matches():
    """Rows whose text equals the query are copies, not retrieval successes.

    79.6% of full-table rows share their embed_text with another row, so counting
    them pinned an earlier version of this metric at 1.0.
    """
    rows = [(f"r{i}", "C1", "q") for i in range(6)]  # all identical to the query
    out = score(rows, k=3, siblings_available=2)
    assert out["matched"] == 0
    assert out["precision"] == 0.0
    assert out["hit"] is False


def test_self_returned_is_detected_by_id():
    rows = [("r0", "C1", "q"), ("r1", "C1", "sib")]
    assert score(rows, k=3, siblings_available=1)["self_returned"] is True


def test_self_not_returned_when_absent():
    rows = [("r5", "C1", "sib")]
    assert score(rows, k=3, siblings_available=1)["self_returned"] is False


def test_zero_available_siblings_does_not_divide_by_zero():
    out = score([("r0", "C1", "q")], k=3, siblings_available=0)
    assert out["slots_available"] == 0
    assert out["precision"] == 0.0


def test_omitting_siblings_available_falls_back_to_returned_count():
    rows = [("r0", "C1", "q"), ("r1", "C1", "a"), ("r2", "C2", "b")]
    out = score(rows, k=3)
    assert out["slots_available"] == 2  # two distinct-text rows returned
    assert out["precision"] == 0.5


def test_overfetch_is_requested_so_duplicates_can_be_discarded():
    client = client_returning([("r0", "C1", "q")])
    score_one(
        client,
        index_name="idx",
        query_text="q",
        query_id="r0",
        query_label="C1",
        id_col="REC_ID",
        label_col="PS_PRODUCT_ID",
        k=3,
        siblings_available=1,
    )
    assert client.api_client.last_body["num_results"] == 3 + OVERFETCH


def test_recall_and_precision_agree_on_whether_anything_was_found():
    """hit is precision > 0; they are computed from one result list."""
    rows = [("r0", "C1", "q"), ("r1", "C1", "sib")]
    out = score(rows, k=3, siblings_available=1)
    assert out["hit"] is True
    assert out["precision"] > 0

    miss = [("r0", "C1", "q"), ("r1", "C9", "other")]
    out2 = score(miss, k=3, siblings_available=1)
    assert out2["hit"] is False
    assert out2["precision"] == 0.0


class TestFilterWireFormat:
    """The filter field name is the whole bug.

    STORAGE_OPTIMIZED endpoints take a SQL predicate in `filter_string`. The query
    REST endpoint silently ignores body keys it does not recognise, so sending
    `filters` or `filters_json` searched the entire corpus while returning HTTP
    200 -- a filtered query that looked like it worked. These tests pin the field
    name so that can never regress quietly.
    """

    def test_filter_is_sent_as_filter_string(self):
        client = client_returning([("r1", "C1", "a")])
        query_index(
            client,
            index_name="idx",
            columns=["REC_ID"],
            query_text="q",
            num_results=5,
            filters={"SOURCE_REGION": "EU"},
        )
        body = client.api_client.last_body
        assert body["filter_string"] == "SOURCE_REGION = 'EU'"
        # The names that silently do nothing must never be used.
        assert "filters" not in body
        assert "filters_json" not in body

    def test_no_filter_key_when_no_filters(self):
        client = client_returning([("r1", "C1", "a")])
        query_index(client, index_name="idx", columns=["REC_ID"], query_text="q", num_results=5)
        assert "filter_string" not in client.api_client.last_body

    def test_single_value_renders_equality(self):
        assert build_filter_string({"SOURCE_REGION": "EU"}) == "SOURCE_REGION = 'EU'"

    def test_list_renders_in_clause(self):
        assert build_filter_string({"SOURCE_REGION": ["EU", "CA"]}) == "SOURCE_REGION IN ('EU', 'CA')"

    def test_multiple_columns_are_anded(self):
        got = build_filter_string({"SOURCE_REGION": "EU", "IMPORT_DOMESTIC": "IMPORT"})
        assert got == "SOURCE_REGION = 'EU' AND IMPORT_DOMESTIC = 'IMPORT'"

    def test_embedded_quote_is_escaped(self):
        """Vendor names contain apostrophes (LEVI'S), which would break the SQL."""
        assert build_filter_string({"VENDOR_NAME": "LEVI'S"}) == "VENDOR_NAME = 'LEVI''S'"

    def test_empty_and_none_values_are_dropped(self):
        assert build_filter_string({"SOURCE_REGION": "", "BANNER": None}) is None
        assert build_filter_string({"SOURCE_REGION": "EU", "BANNER": ""}) == "SOURCE_REGION = 'EU'"

    def test_empty_filters_yield_none(self):
        assert build_filter_string({}) is None

    def test_missing_data_array_is_not_an_error(self):
        """A filter matching nothing omits data_array rather than returning []."""

        class NoRows:
            last_body = None

            def do(self, method, path, body=None):
                return {"manifest": {"columns": [{"name": "REC_ID"}]}, "result": {}}

        cols, rows = query_index(
            SimpleNamespace(api_client=NoRows()),
            index_name="idx",
            columns=["REC_ID"],
            query_text="q",
            num_results=5,
        )
        assert rows == []
        assert cols == ["REC_ID"]


def test_query_type_is_only_sent_when_asked_for():
    """ANN is the index default; passing query_type=None must not force a value."""
    client = client_returning([("r0", "C1", "q")])
    score_one(
        client,
        index_name="idx",
        query_text="q",
        query_id="r0",
        query_label="C1",
        id_col="REC_ID",
        label_col="PS_PRODUCT_ID",
        k=3,
        siblings_available=1,
        query_type=None,
    )
    assert "query_type" not in client.api_client.last_body

    client2 = client_returning([("r0", "C1", "q")])
    score_one(
        client2,
        index_name="idx",
        query_text="q",
        query_id="r0",
        query_label="C1",
        id_col="REC_ID",
        label_col="PS_PRODUCT_ID",
        k=3,
        siblings_available=1,
        query_type="HYBRID",
    )
    assert client2.api_client.last_body["query_type"] == "HYBRID"
