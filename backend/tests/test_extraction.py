"""Clause-matrix extraction: cell semantics and CSV output."""
from __future__ import annotations

import csv
import io

import pytest

from app.extraction.matrix import Cell, MatrixJob, new_job
from app.extraction.schemas import ClauseExtraction


def job_with(cells: list[Cell], doc_ids: list[str], slugs: list[str]) -> MatrixJob:
    job = new_job(doc_ids, slugs, {d: f"Contract {d}" for d in doc_ids})
    for c in cells:
        job.cells[(c.doc_id, c.clause_slug)] = c
    return job


class TestCellDisplay:
    def test_found_shows_the_value(self):
        assert Cell("d1", "governing_law", status="found", value="Delaware").display == "Delaware"

    def test_absent_and_unsure_are_different_symbols(self):
        """A reviewer must be able to tell 'this contract has no cap' from
        'I could not work it out'. Collapsing both to blank destroys that."""
        absent = Cell("d1", "cap_on_liability", status="not_present").display
        unsure = Cell("d1", "cap_on_liability", status="unsure").display
        assert absent == "—"
        assert unsure == "?"
        assert absent != unsure

    def test_error_is_distinct_from_both(self):
        assert Cell("d1", "insurance", status="found", error="boom").display == "error"

    def test_found_without_a_value_still_renders(self):
        assert Cell("d1", "insurance", status="found", value="").display == "found"


class TestSummary:
    def test_counts_each_status(self):
        job = job_with(
            [
                Cell("d1", "a", status="found"),
                Cell("d1", "b", status="not_present"),
                Cell("d2", "a", status="unsure"),
                Cell("d2", "b", status="found", error="x"),
            ],
            ["d1", "d2"], ["a", "b"],
        )
        s = job.summary()
        assert s == {"found": 1, "not_present": 1, "unsure": 1, "error": 1}

    def test_errored_cell_is_not_also_counted_as_found(self):
        """An errored cell carries a default status; it must not inflate 'found'."""
        job = job_with([Cell("d1", "a", status="found", error="timeout")], ["d1"], ["a"])
        assert job.summary()["found"] == 0
        assert job.summary()["error"] == 1


class TestCsv:
    def test_has_a_value_and_source_column_per_clause(self):
        job = job_with([], ["d1"], ["governing_law", "insurance"])
        header = next(csv.reader(io.StringIO(job.to_csv())))
        assert header[0] == "Document"
        assert len(header) == 1 + 2 * 2  # two clauses x (value, source)
        assert any("source" in h for h in header)

    def test_rows_follow_the_requested_document_order(self):
        job = job_with(
            [Cell("d2", "governing_law", status="found", value="Nevada"),
             Cell("d1", "governing_law", status="found", value="Delaware")],
            ["d1", "d2"], ["governing_law"],
        )
        rows = list(csv.reader(io.StringIO(job.to_csv())))[1:]
        assert rows[0][1] == "Delaware"
        assert rows[1][1] == "Nevada"

    def test_source_column_carries_the_citation(self):
        job = job_with(
            [Cell("d1", "governing_law", status="found", value="Delaware",
                  section_path="12 > 12.1", page_number=7, page_confidence="exact")],
            ["d1"], ["governing_law"],
        )
        row = list(csv.reader(io.StringIO(job.to_csv())))[1]
        assert "§ 12 > 12.1" in row[2]
        assert "p.7" in row[2]

    def test_estimated_pages_are_marked_in_the_export(self):
        """The ~ must survive into the spreadsheet, or an estimate reads as fact."""
        job = job_with(
            [Cell("d1", "governing_law", status="found", value="Delaware",
                  page_number=7, page_confidence="estimated")],
            ["d1"], ["governing_law"],
        )
        assert "~p.7" in list(csv.reader(io.StringIO(job.to_csv())))[1][2]

    def test_absent_cells_have_no_source(self):
        job = job_with([Cell("d1", "insurance", status="not_present")], ["d1"], ["insurance"])
        row = list(csv.reader(io.StringIO(job.to_csv())))[1]
        assert row[1] == "—"
        assert row[2] == ""

    def test_missing_cell_produces_empty_columns_not_a_crash(self):
        """A job cancelled halfway must still export what it has."""
        job = job_with([], ["d1"], ["governing_law"])
        row = list(csv.reader(io.StringIO(job.to_csv())))[1]
        assert row == ["Contract d1", "", ""]

    def test_uses_human_clause_labels_not_slugs(self):
        job = job_with([], ["d1"], ["governing_law"])
        header = next(csv.reader(io.StringIO(job.to_csv())))
        assert "Governing Law" in header[1]
        assert "governing_law" not in header[1]

    def test_commas_in_values_are_quoted_correctly(self):
        """A clause value with a comma must not split into two columns."""
        job = job_with(
            [Cell("d1", "governing_law", status="found", value="New York, USA")],
            ["d1"], ["governing_law"],
        )
        row = list(csv.reader(io.StringIO(job.to_csv())))[1]
        assert row[1] == "New York, USA"
        assert len(row) == 3


class TestExtractionSchema:
    def test_status_is_required(self):
        with pytest.raises(Exception):
            ClauseExtraction()

    def test_fields_default_to_empty_so_an_absent_clause_validates(self):
        e = ClauseExtraction(status="not_present")
        assert e.value == "" and e.quote == "" and e.excerpt_index == 0

    def test_rejects_an_invented_status(self):
        with pytest.raises(Exception):
            ClauseExtraction(status="probably")

    def test_schema_is_json_serialisable_for_constrained_decoding(self):
        schema = ClauseExtraction.model_json_schema()
        assert "status" in schema["properties"]
        assert set(schema["properties"]["status"]["enum"]) == {"found", "not_present", "unsure"}


class TestJobLifecycle:
    def test_new_job_records_the_requested_shape(self):
        job = new_job(["d1", "d2"], ["a", "b", "c"], {"d1": "One", "d2": "Two"})
        assert len(job.job_id) == 12
        assert job.status == "queued"
        assert job.doc_ids == ["d1", "d2"]
        assert job.titles["d1"] == "One"

    def test_payload_can_omit_cells_for_cheap_polling(self):
        job = job_with([Cell("d1", "a", status="found")], ["d1"], ["a"])
        assert "cells" not in job.as_dict(include_cells=False)
        assert "cells" in job.as_dict(include_cells=True)

    def test_payload_exposes_human_labels_for_the_ui(self):
        job = new_job(["d1"], ["governing_law"], {"d1": "One"})
        assert job.as_dict()["clause_labels"]["governing_law"] == "Governing Law"
