"""Metadata enrichment (Module 5: Metadata + Tier 1 M1: Metadata Filtering).

Metadata is what turns a semantic search box into a research tool. Without it
every question searches all 510 contracts; with it, "what is the liability cap
in the Accuray agreement?" becomes a filtered search over one document's
liability clauses - which is both faster and dramatically more precise.

Two tiers, deliberately:
  * document-level facts (parties, governing law, contract type, dates) are
    extracted once per document from the opening pages, where they always live;
  * chunk-level tags (clause types, presence of money/dates/durations) are
    derived per chunk by rule.

Both are cheap and deterministic. An optional LLM pass (`enrich_with_llm`) can
fill document-level gaps where the regexes come up empty, but it is off the hot
path - it runs at ingest, once, never per query.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from app.ingestion.chunking import Chunk
from app.ingestion.loaders import LoadedDocument
from app.ingestion.taxonomy import detect_clause_types

logger = logging.getLogger(__name__)

CONTRACT_TYPES = (
    "distributor agreement", "franchise agreement", "joint venture agreement",
    "license agreement", "licensing agreement", "supply agreement", "reseller agreement",
    "service agreement", "services agreement", "consulting agreement", "employment agreement",
    "endorsement agreement", "sponsorship agreement", "co-branding agreement",
    "collaboration agreement", "development agreement", "agency agreement",
    "maintenance agreement", "hosting agreement", "outsourcing agreement",
    "manufacturing agreement", "marketing agreement", "promotion agreement",
    "strategic alliance agreement", "non-compete agreement", "transportation agreement",
    "affiliate agreement", "intellectual property agreement", "master agreement",
    "purchase agreement", "settlement agreement", "confidentiality agreement",
)

US_STATES = (
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky",
    "Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi",
    "Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico",
    "New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming","District of Columbia",
)

ENTITY_SUFFIX = (
    r"(?:Inc|Incorporated|LLC|L\.L\.C|Ltd|Limited|Corp|Corporation|Company|Co|GmbH|AG|"
    r"Aktiengesellschaft|N\.V|B\.V|S\.A|S\.p\.A|PLC|P\.L\.C|LP|L\.P|LLP|Holdings|Group|"
    r"Partnership|Trust|Foundation|Pty|Oy|AB|A/S|SE|KK|Kabushiki Kaisha)\.?"
)

# Party names in contracts are often ALL CAPS ("ACCURAY INCORPORATED"), so the
# suffix match must be case-insensitive - but the name itself must still start
# with a capital, or every sentence fragment becomes a candidate party.
# The suffix is case-insensitive via a *scoped* flag, but the name-start anchor
# stays case-sensitive. A global re.IGNORECASE would make `[A-Z]` match
# lowercase too, turning "a Delaware corporation" into a party.
PARTY_RE = re.compile(
    rf"\b([A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Za-z0-9&.,'\-]+){{0,6}}?\s+(?i:{ENTITY_SUFFIX}))(?=[\s,.;:)]|$)"
)
# "a Delaware corporation" / "an Illinois company" describe a party, they are
# not its name - so a candidate introduced by an article is rejected.
ARTICLE_PREFIX_RE = re.compile(r"\b(?:a|an)\s+$", re.IGNORECASE)
# The preamble that introduces the parties: "...by and between X and Y..."
BETWEEN_RE = re.compile(
    r"\b(?:by and between|between|among)\b(.{0,600}?)(?=\b(?:RECITALS|WITNESSETH|WHEREAS|NOW,? THEREFORE)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
# Defined short names: ACCURAY INCORPORATED ... ("Accuray")
DEFINED_TERM_RE = re.compile(r"\(\s*[\"“]([A-Z][A-Za-z0-9 &.\-]{1,40})[\"”]\s*\)")

# Governing-law detection. The hard part is not finding "laws of X" - it is
# rejecting the many mentions that are NOT the governing-law clause: where a
# party was incorporated, export-control references, "applicable law".
GOVERNING_LAW_RE = re.compile(
    r"(?P<trigger>governed\s+by|construed\s+in\s+accordance\s+with|governing\s+law"
    r"|shall\s+be\s+subject\s+to\s+the\s+laws)"
    r"(?P<middle>[^.]{0,140}?)"
    r"laws?\s+(?:of\s+)?(?:the\s+)?"
    r"(?P<juris>thereof|there\s?of|[A-Z][A-Za-z ]{2,34})",
    re.IGNORECASE,
)
# "governed by the laws thereof" / "the laws of such State" point BACK at a
# jurisdiction named earlier in the sentence ("accepted in the State of Nevada
# and governed by the laws thereof"). Taken literally they yield "Thereof".
ANAPHORIC_JURIS_RE = re.compile(
    r"^(?:thereof|there\s?of|such\s+(?:state|jurisdiction|country|province|commonwealth)"
    r"|that\s+state|this\s+state|said\s+state|the\s+state|such|same)$",
    re.IGNORECASE,
)
# Strips the administrative prefix so "Province Of Ontario" and "Ontario" are
# recognised as the same answer.
JURIS_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:state|commonwealth|province|republic|federal\s+republic|country|"
    r"dominion|kingdom)\s+of\s+",
    re.IGNORECASE,
)
# Used to resolve an anaphoric reference by looking backwards.
NAMED_JURIS_RE = re.compile(
    r"\b(?:State|Commonwealth|Province|Federal\s+Republic|Republic)\s+of\s+([A-Z][A-Za-z ]{2,28})",
)
# If any of these appear just before "laws of", the phrase describes where an
# entity was formed, not what law governs the contract.
INCORPORATION_CONTEXT_RE = re.compile(
    r"\b(?:formed|incorporated|organi[sz]ed|existing|created|duly organized|a corporation|a company|"
    r"a limited liability company|registered)\b[^.]{0,60}$",
    re.IGNORECASE,
)

COUNTRIES = (
    "Germany","France","England","Wales","Scotland","Ireland","Netherlands","Belgium","Switzerland",
    "Austria","Italy","Spain","Portugal","Sweden","Norway","Denmark","Finland","Poland","Japan",
    "China","India","Singapore","Australia","Canada","Israel","Brazil","Mexico","Korea","Bermuda",
    "Luxembourg","Cyprus","Malta","Russia","Turkey","Greece","Hungary","Romania","Czech Republic",
    "Hong Kong","Taiwan","Thailand","Malaysia","Indonesia","Philippines","Vietnam","New Zealand",
    "South Africa","Argentina","Chile","Colombia","United Kingdom","England and Wales",
)

DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)\s+day\s+of\s+[A-Z][a-z]+,?\s+\d{4}\b",
)
MONEY_RE = re.compile(r"(?:US)?\$\s?[\d,]+(?:\.\d{2})?|\b\d[\d,]*\s*(?:dollars|USD|EUR|GBP)\b", re.IGNORECASE)
DURATION_RE = re.compile(r"\b(?:\d{1,4}|one|two|three|five|ten|thirty|sixty|ninety)\s*\(?\d*\)?\s*(?:day|week|month|year)s?\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")

# Words that look like an entity name but never are one.
PARTY_STOPWORDS = {
    "the company", "this company", "such company", "the corporation", "a corporation",
    "the parties", "each party", "either party", "any company", "the group", "no company",
    "and company", "or company", "other company", "its affiliates", "the trust",
}


def infer_contract_type(title: str, head_text: str) -> str:
    """Contract type, title first.

    The title is far more reliable than the body: CUAD filenames carry the type
    ("...-DISTRIBUTOR AGREEMENT"), whereas the body of a distribution agreement
    routinely name-drops other agreements it cross-references. Matching on the
    body first mislabels this contract as a "Strategic Alliance Agreement".
    """
    title_l = title.lower()
    matches = [c for c in CONTRACT_TYPES if c in title_l]
    if matches:
        return max(matches, key=len).title()

    head_l = head_text[:1500].lower()
    matches = [c for c in CONTRACT_TYPES if c in head_l]
    if matches:
        return max(matches, key=len).title()
    return "Agreement"


def _clean_party(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" ,.;:")


def extract_parties(head_text: str, limit: int = 4) -> list[str]:
    """Named entities from the preamble, where parties are always introduced.

    Searched in the "by and between ... " span first, because that is where the
    real parties live; the rest of the preamble is full of addresses, agents and
    cross-referenced entities that are not parties to this contract.
    """
    window = head_text[:6000]
    between = BETWEEN_RE.search(window)
    search_space = between.group(1) if between else window[:2500]

    ordered: list[str] = []
    seen: set[str] = set()
    for match in PARTY_RE.finditer(search_space):
        name = _clean_party(match.group(1))
        key = name.lower()
        if key in seen or key in PARTY_STOPWORDS or len(name) < 4:
            continue
        if key.startswith(("the ", "this ", "such ", "any ", "no ", "and ", "or ", "a ", "an ")):
            continue
        if ARTICLE_PREFIX_RE.search(search_space[max(0, match.start() - 6) : match.start()]):
            continue
        seen.add(key)
        ordered.append(name)
        if len(ordered) >= limit:
            break

    if ordered:
        return ordered

    # Fall back to the defined short names - ("Accuray"), ("Siemens") - which
    # survive even when the formal entity name is formatted unusually.
    defined: list[str] = []
    for match in DEFINED_TERM_RE.finditer(window[:3000]):
        name = _clean_party(match.group(1))
        if name.lower() in {"agreement", "effective date", "the agreement", "parties"}:
            continue
        if name not in defined:
            defined.append(name)
        if len(defined) >= limit:
            break
    return defined


def extract_governing_law(text: str) -> str | None:
    """The jurisdiction whose law governs, or None when we cannot tell.

    Returning None is deliberate. A wrong value here becomes a metadata filter
    that silently hides the right document, which is worse than no filter at
    all - so anything that does not clearly read as a governing-law clause is
    rejected rather than guessed.
    """
    for match in GOVERNING_LAW_RE.finditer(text):
        # Reject "a corporation formed under the laws of Germany" - that is
        # where a party was incorporated, not what governs the agreement.
        preceding = text[max(0, match.start() - 80) : match.start("juris")]
        if INCORPORATION_CONTEXT_RE.search(preceding):
            continue

        raw = _clean_party(match.group("juris"))

        # Resolve a back-reference against the nearest jurisdiction named
        # before it, instead of reporting "Thereof" as the governing law.
        if ANAPHORIC_JURIS_RE.match(raw):
            preceding_named = NAMED_JURIS_RE.findall(text[max(0, match.start() - 400) : match.start()])
            if not preceding_named:
                continue
            raw = preceding_named[-1]

        candidate = JURIS_PREFIX_RE.sub("", raw).strip().title()
        # Trim trailing clause text: "Delaware Without Regard To" -> "Delaware".
        candidate = re.split(
            r"\b(?:And|Without|Which|Shall|For|As|In|Applicable|Excluding|Regardless)\b",
            candidate,
        )[0].strip(" ,.")

        for state in US_STATES:
            if candidate.startswith(state):
                return state
        for country in COUNTRIES:
            if candidate.startswith(country):
                return country
        if 3 <= len(candidate) <= 28 and candidate[:1].isupper() and " Law" not in candidate:
            return candidate
    return None


def extract_dates(text: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    for match in DATE_RE.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def build_document_metadata(doc: LoadedDocument) -> dict[str, Any]:
    """Facts that describe the whole contract - computed once per document."""
    head = doc.full_text[:8000]
    parties = extract_parties(head)
    metadata: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source_path": doc.source_path,
        "file_type": doc.file_type,
        "contract_type": infer_contract_type(doc.title, head),
        "parties": parties,
        "party_primary": parties[0] if parties else None,
        "governing_law": extract_governing_law(doc.full_text),
        "dates": extract_dates(head),
        "page_count": len(doc.pages),
        "char_count": len(doc.full_text),
        "page_confidence": doc.pages[0].confidence if doc.pages else "estimated",
    }
    metadata.update(doc.metadata)
    return {k: v for k, v in metadata.items() if v not in (None, [], "")}


def enrich_chunks(
    chunks: Iterable[Chunk], doc_metadata: dict[str, Any]
) -> list[Chunk]:
    """Attach clause tags and content signals to every chunk.

    The `clause_types_csv` field exists because Chroma's metadata store only
    accepts scalars - a comma-wrapped string lets us do `$contains` filtering on
    a list-valued field.
    """
    enriched: list[Chunk] = []
    for chunk in chunks:
        clause_types = detect_clause_types(chunk.text)
        chunk.metadata.update(
            {
                "title": doc_metadata.get("title", ""),
                "contract_type": doc_metadata.get("contract_type", "Agreement"),
                "party_primary": doc_metadata.get("party_primary"),
                "governing_law": doc_metadata.get("governing_law"),
                "clause_types_csv": "," + ",".join(clause_types) + "," if clause_types else "",
                "clause_type_primary": clause_types[0] if clause_types else None,
                "clause_type_count": len(clause_types),
                "has_money": bool(MONEY_RE.search(chunk.text)),
                "has_date": bool(DATE_RE.search(chunk.text)),
                "has_duration": bool(DURATION_RE.search(chunk.text)),
                "has_percent": bool(PERCENT_RE.search(chunk.text)),
                "word_count": len(chunk.text.split()),
                "section_depth": chunk.section_number.count(".") + 1 if chunk.section_number else 0,
            }
        )
        enriched.append(chunk)
    return enriched


def clause_filter(slugs: list[str]) -> dict[str, Any] | None:
    """Build a Chroma `where` clause for clause-type filtering."""
    if not slugs:
        return None
    if len(slugs) == 1:
        return {"clause_types_csv": {"$contains": f",{slugs[0]},"}}
    return {"$or": [{"clause_types_csv": {"$contains": f",{s},"}} for s in slugs]}
