"""LLM service behavior tests."""

from app.services.llm_service import LLMService


def test_llm_fallback_when_client_errors(monkeypatch) -> None:
    service = LLMService()

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("openrouter unavailable")

    monkeypatch.setattr(service.client, "post", _raise)
    result = service.structure_prescription("Paracetamol 500 mg", ["paracetamol"])

    assert result.confidence_score <= 0.25
    assert result.drugs is not None
    assert result.drugs[0].name == "paracetamol"
