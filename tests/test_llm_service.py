"""Gemini LLM service behavior tests."""

from app.services.llm_service import LLMService, PHARMALINK_SYSTEM_PROMPT


def test_llm_fallback_when_gemini_errors(monkeypatch) -> None:
    service = LLMService()

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("gemini unavailable")

    monkeypatch.setattr(service, "_generate", _raise)
    result = service.structure_prescription("Paracetamol 500 mg", ["paracetamol"])

    assert result.confidence_score <= 0.25
    assert result.drugs is not None
    assert result.drugs[0].name == "paracetamol"


def test_llm_retries_invalid_json(monkeypatch) -> None:
    service = LLMService()
    responses = iter(
        [
            "not json",
            '{"confidence_score":0.9,"description":"Parsed prescription.","drugs":[{"name":"paracetamol","dosage":"500 mg"}]}',
        ]
    )

    monkeypatch.setattr(service, "_generate", lambda *_args, **_kwargs: next(responses))

    result = service.structure_prescription("Paracetamol 500 mg", ["paracetamol"])

    assert result.confidence_score == 0.9
    assert result.drugs is not None
    assert result.drugs[0].name == "paracetamol"


def test_llm_parses_fenced_json() -> None:
    content = """```json
{"confidence_score":0.9,"description":"Parsed prescription.","drugs":[{"name":"paracetamol","dosage":"500 mg"}]}
```"""

    result = LLMService._parse_prescription_response(content)

    assert result.confidence_score == 0.9
    assert result.drugs is not None
    assert result.drugs[0].name == "paracetamol"


def test_chat_prompt_uses_grounded_drugs() -> None:
    service = LLMService()

    prompt = service.build_chat_prompt(
        "Can I take warfrin with ibprofen?", ["Warfarin", "Ibuprofen"]
    )

    assert "* Warfarin" in prompt
    assert "* Ibuprofen" in prompt
    assert "warfrin" in prompt


def test_chat_fallback_returns_safe_response(monkeypatch) -> None:
    service = LLMService()

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("gemini unavailable")

    monkeypatch.setattr(service, "_generate", _raise)

    result = service.answer_chat("Can I take ibuprofen?", ["ibuprofen"])

    assert result.user_query == "Can I take ibuprofen?"
    assert result.normalized_drugs == ["ibuprofen"]
    assert result.warnings
    assert "healthcare professional" in result.response


def test_pharmalink_system_prompt_contains_required_identity() -> None:
    assert PHARMALINK_SYSTEM_PROMPT.startswith("You are PharmaLink Assistant")
    assert "The FAISS index DOES NOT contain:" in PHARMALINK_SYSTEM_PROMPT
    assert "Do not pretend to access live FDA databases." in PHARMALINK_SYSTEM_PROMPT
