"""API tests for prediction endpoint."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.schemas import ChatResponse, PrescriptionResponse
from app.core.config import get_settings


def _client() -> TestClient:
    get_settings().preload_models_on_startup = False
    from app.api import routes

    routes.ocr_service._load_once = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
    routes.normalization_service._load_or_build_faiss = lambda: None  # type: ignore[method-assign]  # noqa: SLF001
    return TestClient(app)


def _build_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color="white")
    buff = BytesIO()
    image.save(buff, format="PNG")
    return buff.getvalue()


def test_predict_success(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes.ocr_service, "extract_text", lambda _: "Paracetamol 500 mg")
    monkeypatch.setattr(
        routes.extraction_service, "extract_candidates", lambda _: ["paracetamol"]
    )
    monkeypatch.setattr(
        routes.normalization_service, "normalize", lambda _: ["paracetamol"]
    )
    monkeypatch.setattr(
        routes.llm_service,
        "structure_prescription",
        lambda _text, _drugs: PrescriptionResponse(
            confidence_score=0.92,
            description="Likely pain management prescription.",
            drugs=[{"name": "paracetamol", "dosage": "500 mg"}],
        ),
    )

    client = _client()
    response = client.post(
        "/predict",
        files={"image": ("rx.png", _build_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_drugs"] == ["paracetamol"]
    assert payload["result"]["confidence_score"] == 0.92


def test_root_health_endpoint() -> None:
    client = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_non_image() -> None:
    client = _client()
    response = client.post(
        "/predict",
        files={"image": ("rx.txt", b"not image", "text/plain")},
    )
    assert response.status_code == 400


def test_chat_success(monkeypatch) -> None:
    from app.api import routes

    monkeypatch.setattr(
        routes.extraction_service,
        "extract_candidates",
        lambda text: ["ibuprofen", "warfarin"],
    )
    monkeypatch.setattr(
        routes.normalization_service,
        "normalize",
        lambda candidates: ["Ibuprofen", "Warfarin"],
    )
    monkeypatch.setattr(
        routes.llm_service,
        "answer_chat",
        lambda message, drugs: ChatResponse(
            user_query=message,
            normalized_drugs=drugs,
            response="Ibuprofen and Warfarin may interact and should be reviewed.",
            warnings=["Informational support only."],
        ),
    )

    client = _client()
    response = client.post(
        "/chat",
        json={"message": "Can I take ibuprofen with warfarin?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_drugs"] == ["Ibuprofen", "Warfarin"]
    assert "may interact" in payload["response"]


def test_chat_rejects_empty_message() -> None:
    client = _client()
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422
