"""The 41 CUAD clause categories, as a machine-usable taxonomy.

Metadata is only useful if it is a *closed* vocabulary - a filter for
"termination" works only when every termination clause carries the same tag.
These are the exact 41 categories Atticus Project lawyers annotated, which
means the same vocabulary drives three things at once: metadata filtering at
query time, the clause-type facets in the UI, and the ground-truth join in the
evaluation harness.

Each entry carries regex patterns for cheap deterministic tagging at ingest
time. The rules are high-precision, not high-recall: it is better to leave a
clause untagged (dense retrieval still finds it) than to mislabel it and have a
metadata filter hide the right answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClauseCategory:
    slug: str
    cuad_name: str      # exact CUAD column name - the evaluation join key
    label: str          # human-friendly, used in the UI
    description: str
    patterns: tuple[str, ...]


CATEGORIES: tuple[ClauseCategory, ...] = (
    ClauseCategory("document_name", "Document Name", "Document Name",
        "The name of the contract.",
        (r"\bagreement\b.{0,40}\bentered into\b", r"^\s*[A-Z][A-Z \-]{8,80}AGREEMENT\s*$")),
    ClauseCategory("parties", "Parties", "Parties",
        "The two or more parties who signed the contract.",
        (r"\bby and between\b", r"\bbetween\b.{0,80}\band\b.{0,80}\b(Inc|LLC|Ltd|Corp|GmbH|S\.A\.)\b",
         r"\bhereinafter referred to as\b")),
    ClauseCategory("agreement_date", "Agreement Date", "Agreement Date",
        "The date the contract was signed.",
        (r"\bdated as of\b", r"\bthis\s+\d{1,2}(st|nd|rd|th)?\s+day of\b", r"\bexecuted on\b")),
    ClauseCategory("effective_date", "Effective Date", "Effective Date",
        "The date when the contract becomes effective.",
        (r"\beffective date\b", r"\bshall become effective\b", r"\bcommences? (?:on|as of)\b")),
    ClauseCategory("expiration_date", "Expiration Date", "Expiration Date",
        "When the initial term expires.",
        (r"\bexpir(?:e|es|ation)\b", r"\binitial term\b", r"\bshall (?:continue|remain in effect) until\b")),
    ClauseCategory("renewal_term", "Renewal Term", "Renewal Term",
        "The renewal term after the initial term expires.",
        (r"\brenewal term\b", r"\bautomatically renew\w*\b", r"\bsuccessive\s+(?:one|two|three|\d+)[\s\-]*\(?\d*\)?\s*year\b")),
    ClauseCategory("notice_to_terminate_renewal", "Notice Period To Terminate Renewal", "Notice to Terminate Renewal",
        "Notice period required to stop an automatic renewal.",
        (r"\bnotice\b.{0,60}\b(?:not to renew|non-?renewal)\b", r"\b(?:thirty|sixty|ninety|\d{1,3})\s*\(?\d*\)?\s*days.{0,40}\bprior to\b.{0,40}\bexpiration\b")),
    ClauseCategory("governing_law", "Governing Law", "Governing Law",
        "Which jurisdiction's law governs the contract.",
        (r"\bgoverned by\b.{0,60}\blaws?\b", r"\bgoverning law\b", r"\bconstrued in accordance with the laws\b")),
    ClauseCategory("most_favored_nation", "Most Favored Nation", "Most Favored Nation",
        "Better terms given to a third party must be extended to this counterparty.",
        (r"\bmost favou?red (?:nation|customer)\b", r"\bno less favou?rable\b", r"\bterms.{0,30}no less favou?rable than\b")),
    ClauseCategory("non_compete", "Non-Compete", "Non-Compete",
        "Restriction on a party competing with the counterparty.",
        (r"\bnon-?compet\w+\b", r"\bshall not\b.{0,60}\bcompete\b", r"\bcompeting (?:product|business|activity)\b")),
    ClauseCategory("exclusivity", "Exclusivity", "Exclusivity",
        "Exclusive dealing commitment.",
        (r"\bexclusiv\w+\b", r"\bsole and exclusive\b", r"\bshall be the (?:sole|exclusive)\b")),
    ClauseCategory("no_solicit_customers", "No-Solicit Of Customers", "No-Solicit of Customers",
        "Restriction on soliciting the counterparty's customers.",
        (r"\bnot\b.{0,40}\bsolicit\b.{0,60}\b(?:customer|client|partner)s?\b", r"\bnon-?solicitation\b.{0,60}\bcustomers?\b")),
    ClauseCategory("competitive_restriction_exception", "Competitive Restriction Exception", "Competitive Restriction Exception",
        "Carve-outs to non-compete / exclusivity / no-solicit.",
        (r"\b(?:except|exception|carve-?out|notwithstanding)\b.{0,80}\b(?:non-?compete|exclusiv\w+|solicit)\w*\b",
         r"\bshall not (?:apply|prohibit|restrict)\b")),
    ClauseCategory("no_solicit_employees", "No-Solicit Of Employees", "No-Solicit of Employees",
        "Restriction on soliciting or hiring the counterparty's employees.",
        (r"\bnot\b.{0,40}\b(?:solicit|hire|employ)\b.{0,60}\bemployees?\b", r"\bno-?solicit\w*\b.{0,40}\bemployees?\b")),
    ClauseCategory("non_disparagement", "Non-Disparagement", "Non-Disparagement",
        "Requirement not to disparage the counterparty.",
        (r"\bdisparag\w+\b", r"\bnot.{0,30}make any.{0,30}(?:negative|derogatory) statements?\b")),
    ClauseCategory("termination_for_convenience", "Termination For Convenience", "Termination for Convenience",
        "Termination without cause, on notice.",
        (r"\bterminate\b.{0,60}\b(?:for convenience|without cause|for any reason|at any time)\b",
         r"\btermination for convenience\b", r"\bwith or without cause\b")),
    ClauseCategory("rofr_rofo_rofn", "Rofr/Rofo/Rofn", "Right of First Refusal / Offer",
        "Right of first refusal, offer, or negotiation.",
        (r"\bright of first (?:refusal|offer|negotiation)\b", r"\bROFR\b|\bROFO\b|\bROFN\b")),
    ClauseCategory("change_of_control", "Change Of Control", "Change of Control",
        "Rights triggered by a change of control of a party.",
        (r"\bchange (?:of|in) control\b", r"\bmerger(?:,| or) (?:consolidation|acquisition)\b",
         r"\backquisition of (?:all or substantially all|a majority)\b")),
    ClauseCategory("anti_assignment", "Anti-Assignment", "Anti-Assignment",
        "Consent or notice required to assign the contract.",
        (r"\bassign\w*\b.{0,80}\b(?:prior written consent|not.{0,20}without)\b",
         r"\bshall not (?:be assigned|assign)\b", r"\bno(?:t|n)-?assignment\b")),
    ClauseCategory("revenue_profit_sharing", "Revenue/Profit Sharing", "Revenue / Profit Sharing",
        "Obligation to share revenue or profit.",
        (r"\b(?:revenue|profit|royalt\w+)\b.{0,40}\bshar\w+\b", r"\b\d{1,3}\s*%\s*of\s*(?:net|gross)\b")),
    ClauseCategory("price_restrictions", "Price Restrictions", "Price Restrictions",
        "Restriction on raising or lowering prices.",
        (r"\bprice\b.{0,50}\b(?:increase|decrease|shall not)\b", r"\bnot.{0,30}(?:raise|increase).{0,30}price\b")),
    ClauseCategory("minimum_commitment", "Minimum Commitment", "Minimum Commitment",
        "Minimum order size or purchase volume.",
        (r"\bminimum\b.{0,40}\b(?:purchase|order|quantit|commitment|volume|amount)\w*\b",
         r"\bshall (?:purchase|order)\b.{0,40}\bat least\b")),
    ClauseCategory("volume_restriction", "Volume Restriction", "Volume Restriction",
        "Fee increase or consent required beyond a usage threshold.",
        (r"\bexceeds?\b.{0,50}\b(?:threshold|limit|volume|number of (?:users|seats|copies))\b",
         r"\badditional (?:fee|charge)s?\b.{0,50}\bexceed\b")),
    ClauseCategory("ip_ownership_assignment", "Ip Ownership Assignment", "IP Ownership Assignment",
        "IP created by one party becomes the counterparty's property.",
        (r"\b(?:hereby )?assigns?\b.{0,60}\b(?:intellectual property|right, title and interest)\b",
         r"\bwork(?:s)? made for hire\b", r"\bshall be the (?:sole and exclusive )?property of\b")),
    ClauseCategory("joint_ip_ownership", "Joint Ip Ownership", "Joint IP Ownership",
        "Jointly owned intellectual property.",
        (r"\bjointly owned?\b", r"\bjoint(?:ly)? (?:own|develop)\w*\b.{0,40}\b(?:intellectual property|invention)\b")),
    ClauseCategory("license_grant", "License Grant", "License Grant",
        "A licence granted by one party to the other.",
        (r"\bhereby grants?\b.{0,60}\blicen[cs]e\b", r"\bgrants? to\b.{0,60}\ba\b.{0,40}\blicen[cs]e\b")),
    ClauseCategory("non_transferable_license", "Non-Transferable License", "Non-Transferable Licence",
        "Licence that cannot be transferred to a third party.",
        (r"\bnon-?transferable\b", r"\bnon-?sublicensable\b", r"\bmay not (?:transfer|sublicen[cs]e)\b")),
    ClauseCategory("affiliate_license_licensor", "Affiliate License-Licensor", "Affiliate Licence (Licensor)",
        "Licence granted by affiliates of the licensor.",
        (r"\baffiliates?\b.{0,60}\bgrants?\b.{0,40}\blicen[cs]e\b", r"\blicensor(?:'s)? affiliates?\b")),
    ClauseCategory("affiliate_license_licensee", "Affiliate License-Licensee", "Affiliate Licence (Licensee)",
        "Licence extended to affiliates of the licensee.",
        (r"\blicensee(?:'s)? affiliates?\b", r"\band its affiliates?\b.{0,40}\blicen[cs]e\b")),
    ClauseCategory("unlimited_license", "Unlimited/All-You-Can-Eat-License", "Unlimited / Enterprise Licence",
        "Enterprise, all-you-can-eat, or unlimited usage licence.",
        (r"\bunlimited\b.{0,40}\b(?:licen[cs]e|use|users|copies)\b", r"\benterprise[- ]wide licen[cs]e\b",
         r"\ball[- ]you[- ]can[- ]eat\b")),
    ClauseCategory("irrevocable_perpetual_license", "Irrevocable Or Perpetual License", "Irrevocable / Perpetual Licence",
        "Licence that is irrevocable or perpetual.",
        (r"\b(?:irrevocable|perpetual)\b.{0,50}\blicen[cs]e\b", r"\blicen[cs]e\b.{0,50}\b(?:irrevocable|perpetual)\b")),
    ClauseCategory("source_code_escrow", "Source Code Escrow", "Source Code Escrow",
        "Source code deposited into escrow.",
        (r"\bescrow\b", r"\bsource code\b.{0,60}\bdeposit\w*\b")),
    ClauseCategory("post_termination_services", "Post-Termination Services", "Post-Termination Services",
        "Obligations that survive termination or expiry.",
        (r"\bsurviv\w+\b.{0,60}\b(?:termination|expiration)\b", r"\b(?:upon|after|following) (?:termination|expiration)\b.{0,80}\bshall\b",
         r"\bwind[- ]down\b", r"\btransition (?:services|assistance)\b")),
    ClauseCategory("audit_rights", "Audit Rights", "Audit Rights",
        "Right to audit the counterparty's books or premises.",
        (r"\baudit\w*\b", r"\bright to (?:inspect|examine)\b.{0,50}\b(?:books|records)\b")),
    ClauseCategory("uncapped_liability", "Uncapped Liability", "Uncapped Liability",
        "Liability that is not capped.",
        (r"\bunlimited liability\b", r"\bno(?:t)? (?:be )?(?:subject to|limited by) (?:any )?(?:cap|limitation)\b",
         r"\bexclusions? (?:from|to) (?:the )?limitation of liability\b")),
    ClauseCategory("cap_on_liability", "Cap On Liability", "Cap on Liability",
        "Limit on a party's liability.",
        (r"\blimitation of liability\b", r"\b(?:shall not|in no event).{0,60}\bexceed\b",
         r"\baggregate liability\b", r"\bliability.{0,40}\b(?:cap|capped)\b")),
    ClauseCategory("liquidated_damages", "Liquidated Damages", "Liquidated Damages",
        "Pre-agreed damages or fee on breach.",
        (r"\bliquidated damages\b", r"\bpenalt(?:y|ies)\b.{0,50}\bbreach\b", r"\bearly termination fee\b")),
    ClauseCategory("warranty_duration", "Warranty Duration", "Warranty Duration",
        "How long a warranty lasts.",
        (r"\bwarrant(?:y|ies)\b.{0,60}\b(?:period|for a period of|days|months|years)\b",
         r"\bwarranty period\b")),
    ClauseCategory("insurance", "Insurance", "Insurance",
        "Insurance a party must maintain.",
        (r"\binsurance\b", r"\bshall (?:maintain|carry|procure)\b.{0,50}\b(?:coverage|polic(?:y|ies))\b")),
    ClauseCategory("covenant_not_to_sue", "Covenant Not To Sue", "Covenant Not to Sue",
        "Restriction on challenging IP validity, or release of claims.",
        (r"\bcovenant not to sue\b", r"\brelease of claims\b", r"\bshall not (?:challenge|contest)\b.{0,50}\bvalidity\b")),
    ClauseCategory("third_party_beneficiary", "Third Party Beneficiary", "Third-Party Beneficiary",
        "A non-contracting party who can enforce the contract.",
        (r"\bthird[- ]party beneficiar(?:y|ies)\b", r"\bno third party.{0,40}\brights?\b")),
)

BY_SLUG: dict[str, ClauseCategory] = {c.slug: c for c in CATEGORIES}
BY_CUAD_NAME: dict[str, ClauseCategory] = {c.cuad_name: c for c in CATEGORIES}

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    c.slug: tuple(re.compile(p, re.IGNORECASE) for p in c.patterns) for c in CATEGORIES
}


def detect_clause_types(text: str, max_tags: int = 6) -> list[str]:
    """Rule-based clause tagging - deterministic, free, and runs at ingest time.

    Scored by how many distinct patterns fire, so a chunk that merely mentions
    "insurance" in passing loses to one that actually is the insurance clause.
    """
    scores: list[tuple[int, str]] = []
    for slug, patterns in _COMPILED.items():
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scores.append((hits, slug))
    scores.sort(key=lambda t: (-t[0], t[1]))
    return [slug for _, slug in scores[:max_tags]]


def category_labels() -> list[dict[str, str]]:
    """Facet list for the UI filter panel."""
    return [{"slug": c.slug, "label": c.label, "description": c.description} for c in CATEGORIES]
