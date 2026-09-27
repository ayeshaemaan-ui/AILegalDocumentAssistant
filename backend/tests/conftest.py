"""Shared fixtures.

Tests run against a temporary data directory so they never touch the developer's
real index, and never need network or an API key. Anything that would call a
model is either stubbed or excluded.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture(scope="session")
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.fixture(scope="session")
def settings(tmp_data_dir):
    from app.config import Settings

    return Settings(
        environment="test",
        chroma_dir=tmp_data_dir / "chroma",
        state_db_path=tmp_data_dir / "state.db",
        trace_db_path=tmp_data_dir / "traces.db",
        upload_dir=tmp_data_dir / "uploads",
        anthropic_api_key="test-key-not-used",
    )


@pytest.fixture
def sample_contract() -> str:
    """A miniature contract exercising the structures the parser looks for."""
    return """DISTRIBUTOR AGREEMENT

This Distributor Agreement ("Agreement") is entered into by and between ACME ROBOTICS INC., a Delaware corporation ("Supplier"), and NORDIC DISTRIBUTION AB, a company organised under the laws of Sweden ("Distributor"), as of March 14, 2021 ("Effective Date").

1. DEFINITIONS

1.1. "Products" means the robotic systems manufactured by Supplier.

1.2. "Territory" means the Nordic region.

2. APPOINTMENT

2.1. Appointment. Supplier appoints Distributor as its exclusive distributor of the Products in the Territory.

2.2. Minimum Commitment. Distributor shall purchase at least 500 units per calendar year.

3. TERM AND TERMINATION

3.1. Term. This Agreement commences on the Effective Date and continues for three (3) years, renewing automatically for successive one (1) year periods unless either party gives ninety (90) days written notice prior to expiration.

3.2. Termination for Convenience. Supplier may terminate this Agreement at any time upon one hundred eighty (180) days written notice to Distributor.

4. LIABILITY

4.1. Limitation of Liability. In no event shall either party's aggregate liability exceed the total fees paid under this Agreement in the twelve (12) months preceding the claim.

4.2. Insurance. Each party shall maintain commercial general liability insurance of not less than $2,000,000 per occurrence.

5. GOVERNING LAW

5.1. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.
"""


@pytest.fixture
def loaded_document(sample_contract, tmp_path):
    from app.ingestion.loaders import load_document

    path = tmp_path / "ACME_ROBOTICS-DISTRIBUTOR AGREEMENT.txt"
    path.write_text(sample_contract, encoding="utf-8")
    return load_document(path)
