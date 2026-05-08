"""Normalization service tests with mocked retrieval layers."""

from app.services.normalization_service import NormalizationService


def test_normalization_combines_scores(monkeypatch) -> None:
    service = NormalizationService()

    monkeypatch.setattr(service, "_load_or_build_faiss", lambda: None)
    service.drug_names = ["paracetamol", "amoxicillin", "ibuprofen"]

    monkeypatch.setattr(
        service,
        "_fuzzy_candidates",
        lambda _token, limit=5: {"paracetamol": 0.9, "ibuprofen": 0.5},
    )
    monkeypatch.setattr(
        service,
        "_embedding_candidates",
        lambda _token, limit=5: {"paracetamol": 0.8, "amoxicillin": 0.7},
    )

    result = service.normalize(["paracitamol"], top_k=2)

    assert result[0] == "paracetamol"
    assert len(result) == 2
