"""Document processing, chunking and metadata extraction."""
from __future__ import annotations

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_document, make_doc_id, normalise_text
from app.ingestion.metadata import (
    build_document_metadata,
    enrich_chunks,
    extract_governing_law,
    extract_parties,
    infer_contract_type,
)
from app.ingestion.taxonomy import BY_SLUG, CATEGORIES, detect_clause_types


class TestLoaders:
    def test_loads_text_and_assigns_pages(self, loaded_document):
        assert loaded_document.full_text
        assert loaded_document.pages
        assert loaded_document.pages[0].confidence in {"exact", "recovered", "estimated"}

    def test_doc_id_is_content_addressed(self, tmp_path):
        """Same content under a different filename must collide, so re-uploading
        a file cannot silently duplicate it in the index."""
        text = "This Agreement is governed by Delaware law."
        a = make_doc_id(tmp_path / "one.txt", text)
        b = make_doc_id(tmp_path / "one.txt", text)
        c = make_doc_id(tmp_path / "one.txt", text + " Amended.")
        assert a == b
        assert a != c

    def test_normalise_preserves_content(self):
        raw = "Section  1.\n\n\n\nThe  indemni-\nfication clause “applies”."
        out = normalise_text(raw)
        assert "indemnification" in out   # de-hyphenated across the line break
        assert '"applies"' in out         # smart quotes flattened
        assert "Section 1." in out

    def test_rejects_unsupported_type(self, tmp_path):
        path = tmp_path / "contract.xyz"
        path.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(path)

    def test_page_lookup_by_offset(self, loaded_document):
        page = loaded_document.page_for_offset(10)
        assert page is not None
        assert page.char_start <= 10 < page.char_end


class TestChunking:
    @pytest.mark.parametrize("strategy", ["recursive", "legal_structural"])
    def test_produces_chunks(self, loaded_document, settings, strategy):
        chunks = chunk_document(loaded_document, settings, strategy=strategy)
        assert chunks
        assert all(c.text.strip() for c in chunks)
        assert all(c.doc_id == loaded_document.doc_id for c in chunks)

    def test_structural_chunking_captures_section_paths(self, loaded_document, settings):
        chunks = chunk_document(loaded_document, settings, strategy="legal_structural")
        with_paths = [c for c in chunks if c.section_path]
        assert with_paths, "structural chunking should attach section paths"
        assert any("3" in (c.section_number or "") for c in chunks)

    def test_chunk_ids_are_unique(self, loaded_document, settings):
        chunks = chunk_document(loaded_document, settings, strategy="legal_structural")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_offsets_point_into_the_document(self, loaded_document, settings):
        for chunk in chunk_document(loaded_document, settings, strategy="legal_structural"):
            assert 0 <= chunk.char_start <= len(loaded_document.full_text)

    def test_metadata_is_flat_for_the_vector_store(self, loaded_document, settings):
        """Chroma rejects nested metadata values, so to_metadata must be scalar."""
        chunk = chunk_document(loaded_document, settings, strategy="legal_structural")[0]
        for value in chunk.to_metadata().values():
            assert not isinstance(value, (dict, list))

    def test_unknown_strategy_is_rejected(self, loaded_document, settings):
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            chunk_document(loaded_document, settings, strategy="nonexistent")


class TestMetadata:
    def test_contract_type_prefers_the_title(self, loaded_document):
        """The body name-drops other agreements; the title is authoritative."""
        assert "Distributor" in infer_contract_type(loaded_document.title, loaded_document.full_text)

    def test_extracts_both_parties(self, loaded_document):
        parties = extract_parties(loaded_document.full_text[:8000])
        joined = " ".join(parties).upper()
        assert "ACME" in joined
        assert "NORDIC" in joined

    def test_rejects_article_prefixed_descriptions(self, loaded_document):
        """"a Delaware corporation" describes a party, it is not one."""
        parties = extract_parties(loaded_document.full_text[:8000])
        assert not any(p.lower().startswith(("a ", "an ")) for p in parties)

    def test_governing_law(self, loaded_document):
        assert extract_governing_law(loaded_document.full_text) == "Delaware"

    def test_ignores_incorporation_jurisdiction(self):
        """"organised under the laws of Sweden" is where a party was formed."""
        text = "NORDIC AB, a company organised under the laws of Sweden, agrees as follows."
        assert extract_governing_law(text) is None

    def test_resolves_anaphoric_reference(self):
        """"the laws thereof" points back at a jurisdiction named earlier."""
        text = (
            "This Agreement is accepted by Company in the State of Nevada and shall be "
            "governed by and construed in accordance with the laws thereof."
        )
        assert extract_governing_law(text) == "Nevada"

    def test_document_metadata_has_no_empty_values(self, loaded_document):
        meta = build_document_metadata(loaded_document)
        assert meta["doc_id"] == loaded_document.doc_id
        assert all(v not in (None, [], "") for v in meta.values())


class TestTaxonomy:
    def test_has_all_41_cuad_categories(self):
        assert len(CATEGORIES) == 41
        assert len({c.slug for c in CATEGORIES}) == 41

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("This Agreement shall be governed by the laws of Delaware.", "governing_law"),
            ("In no event shall liability exceed the fees paid.", "cap_on_liability"),
            ("Supplier may terminate this Agreement at any time upon notice.", "termination_for_convenience"),
            ("Each party shall maintain commercial general liability insurance.", "insurance"),
            ("Distributor shall purchase at least 500 units per year.", "minimum_commitment"),
        ],
    )
    def test_detects_expected_clause_type(self, text, expected):
        assert expected in detect_clause_types(text)

    def test_tags_are_from_the_closed_vocabulary(self, loaded_document, settings):
        """A filter only works if every tag comes from the known taxonomy."""
        chunks = enrich_chunks(
            chunk_document(loaded_document, settings, strategy="legal_structural"),
            build_document_metadata(loaded_document),
        )
        for chunk in chunks:
            for slug in chunk.metadata["clause_types_csv"].strip(",").split(","):
                if slug:
                    assert slug in BY_SLUG

    def test_returns_nothing_for_unrelated_text(self):
        assert detect_clause_types("The weather today is pleasant and mild.") == []
