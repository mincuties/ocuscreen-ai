import io
import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import backend.app as api
import backend.image_predictor as ip
from backend.risk_engine import assess_risk


def prediction(label="Normal", status="valid"):
    return {
        "label": label,
        "status": status,
        "class_id": ip.EXPECTED.index(label),
        "confidence": 0.9,
        "message": "Test",
    }


def photo(color="white"):
    b = io.BytesIO()
    Image.new("RGB", (300, 240), color).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_artifacts", lambda: None)
    monkeypatch.setattr(api, "predict_image", lambda b, lang="en": prediction())
    with TestClient(api.app) as c:
        yield c


@pytest.mark.parametrize(
    "label,status,symptoms,level",
    [
        ("Normal", "valid", [], "low"),
        ("Normal", "valid", ["itch"], "medium"),
        ("Normal", "valid", ["vision_loss"], "high"),
        ("Normal", "valid", ["contacts", "red_eye"], "high"),
        ("Back Ground", "invalid", [], None),
        ("Back Ground", "invalid", ["severe_pain"], "high"),
        ("Redness", "uncertain", [], None),
        ("Redness", "valid", [], "medium"),
        ("Yellow Eye", "valid", [], "high"),
        ("Normal", "valid", ["yellow_skin"], "high"),
    ],
)
def test_risk(label, status, symptoms, level):
    assert assess_risk(prediction(label, status), symptoms)["level"] == level


def test_labels():
    labels = ip.read_labels()
    assert labels[10] == (10, "Pale Inner Eyelid")
    assert labels[11] == (11, "Back Ground")


def test_preprocess():
    a = ip.preprocess(photo())
    assert a.shape == (1, 224, 224, 3) and a.dtype == np.float32
    assert np.all(a == 1)
    assert np.all(ip.preprocess(photo("black")) == -1)
    with pytest.raises(ip.InvalidImage):
        ip.preprocess(b"not an image")


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_preprocess_rejects_other_formats(fmt):
    b = io.BytesIO()
    Image.new("RGB", (300, 240), "white").save(b, format=fmt)
    with pytest.raises(ip.InvalidImage):
        ip.preprocess(b.getvalue())


def test_predict_background(monkeypatch):
    class Model:
        def __call__(self, batch, training=False):
            a = np.zeros((1, 12))
            a[0, 11] = 1
            return a

    monkeypatch.setattr(ip, "load_artifacts", lambda: (Model(), ip.read_labels()))
    assert ip.predict_image(photo())["status"] == "invalid"


def test_routes(client):
    assert client.get("/").status_code == 200
    assert len(client.get("/api/symptoms").json()["symptoms"]) == 28
    assert client.get("/model/labels.txt").status_code == 404
    r = client.post(
        "/api/screen",
        files={"image": ("eye.png", photo(), "image/png")},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.status_code == 200 and r.json()["risk"]["level"] == "high"


@pytest.mark.parametrize("symptoms", ["not-json", "{}", '["fake"]', "[1]"])
def test_bad_symptoms(client, symptoms):
    assert (
        client.post(
            "/api/screen",
            files={"image": ("e.png", photo())},
            data={"symptoms": symptoms},
        ).status_code
        == 422
    )


def test_invalid_preserves_warning(client, monkeypatch):
    def invalid(b, lang="en"):
        raise ip.InvalidImage("image_unreadable")

    monkeypatch.setattr(api, "predict_image", invalid)
    r = client.post(
        "/api/screen",
        files={"image": ("e.png", b"bad")},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.json()["risk"]["level"] == "high"
    assert r.json()["image"]["status"] == "invalid"


def test_missing_model_preserves_warning(client, monkeypatch):
    def missing(b, lang="en"):
        raise ip.ModelUnavailable("missing")

    monkeypatch.setattr(api, "predict_image", missing)
    r = client.post(
        "/api/screen",
        files={"image": ("e.png", photo())},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.status_code == 503 and r.json()["risk"]["level"] == "high"


def test_limits(client):
    assert (
        client.post(
            "/api/screen",
            content=b"x" * (ip.MAX_BYTES + 65537),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        ).status_code
        == 413
    )
    api.app.state.busy = True
    assert client.post("/api/screen").status_code == 429
    api.app.state.busy = False


def test_inference_failure_is_model_unavailable(monkeypatch):
    class BrokenModel:
        def __call__(self, batch, training=False):
            raise RuntimeError("device unavailable")

    monkeypatch.setattr(ip, "load_artifacts", lambda: (BrokenModel(), ip.read_labels()))
    with pytest.raises(ip.ModelUnavailable):
        ip.predict_image(photo())


def test_frontend_assets_and_cache_policy(client):
    for path in [
        "/js/app.js",
        "/js/components.js",
        "/js/api.js",
        "/js/validation.js",
        "/css/styles.css",
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert "must-revalidate" in response.headers["cache-control"]
    assert client.get("/api/symptoms").headers["cache-control"] == "no-store"
    combinations = client.get("/api/symptoms").json()["urgent_combinations"]
    assert [c["id"] for c in combinations] == [
        c["id"] for c in api.KB["urgent_combinations"]
    ]
    assert all(isinstance(c["reason"], str) for c in combinations)


def test_rejected_request_releases_inference_slot(client):
    response = client.post(
        "/api/screen", content=b"bad", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 415
    assert not api.app.state.busy
    response = client.post("/api/screen", files={"image": ("eye.png", photo())})
    assert response.status_code == 200


def test_malformed_multipart_releases_slot(client):
    response = client.post(
        "/api/screen", content=b"bad", headers={"Content-Type": "multipart/form-data"}
    )
    assert response.status_code == 400
    assert not api.app.state.busy


def test_duplicate_symptom_fields_rejected(client):
    response = client.post('/api/screen', files=[
        ('image', ('eye.png', photo(), 'image/png')),
        ('symptoms', (None, '["itch"]')),
        ('symptoms', (None, '[]')),
    ])
    assert response.status_code == 422
    assert not api.app.state.busy


def test_declared_oversize_rejected_before_body(client):
    response = client.post('/api/screen', content=b'partial', headers={
        'Content-Type': 'multipart/form-data; boundary=test',
        'Content-Length': str(ip.MAX_BYTES + 65537),
    })
    assert response.status_code == 413
    assert not api.app.state.busy


def test_knowledge_base_has_both_languages():
    def texts(node):
        if isinstance(node, dict):
            if set(node) == {"en", "vi"}:
                yield node
            else:
                for value in node.values():
                    yield from texts(value)
        elif isinstance(node, list):
            for value in node:
                yield from texts(value)

    pairs = list(texts(api.KB))
    # 28 symptoms + 11 image signs + 5 combinations + 9 patterns x 2 + 4 sources + 2
    assert len(pairs) == 68
    assert all(p["en"].strip() and p["vi"].strip() for p in pairs)


def test_risk_language_defaults_to_english():
    pred = prediction("Yellow Eye")
    en = assess_risk(pred, ["vision_loss"])
    vi = assess_risk(pred, ["vision_loss"], "vi")
    assert en["level"] == vi["level"] == "high"
    assert en["label"] == "High" and vi["label"] == "Cao"
    assert "Sudden reduced or lost vision" in en["reasons"]
    assert "Giảm / mất thị lực đột ngột" in vi["reasons"]
    assert en["candidates"][0]["name"] == "Yellow eyes needing a general check-up"
    assert vi["candidates"][0]["name"] == "Vàng mắt cần kiểm tra toàn thân"
    assert assess_risk(prediction("Back Ground", "invalid"), [])["label"] == "Not assessed"


def test_symptoms_endpoint_language(client):
    en = client.get("/api/symptoms").json()
    vi = client.get("/api/symptoms?lang=vi").json()
    fallback = client.get("/api/symptoms?lang=fr").json()
    assert en["lang"] == fallback["lang"] == "en" and vi["lang"] == "vi"
    assert en["symptoms"][1] == {"id": "itch", "label": "Itchy eyes"}
    assert vi["symptoms"][1] == {"id": "itch", "label": "Ngứa mắt"}
    assert fallback["symptoms"] == en["symptoms"]
    assert en["disclaimer"].startswith("Experimental")


def test_screen_errors_follow_language(client, monkeypatch):
    def invalid(b, lang="en"):
        raise ip.InvalidImage("image_format")

    monkeypatch.setattr(api, "predict_image", invalid)
    files = {"image": ("e.png", b"bad")}
    en = client.post("/api/screen", files=files).json()
    vi = client.post("/api/screen?lang=vi", files=files).json()
    assert en["image"]["message"] == "Only JPEG, PNG or WebP images are accepted."
    assert vi["image"]["message"] == "Chỉ nhận ảnh JPEG, PNG hoặc WebP."
    assert en["risk"]["label"] == "Not assessed" and vi["risk"]["label"] == "Chưa đánh giá"
    r = client.post("/api/screen?lang=vi", files=files, data={"symptoms": "x"})
    assert r.status_code == 422
    assert r.json()["detail"] == "symptoms phải là chuỗi JSON dạng danh sách."
    r = client.post("/api/screen", files=files, data={"symptoms": "x"})
    assert r.json()["detail"] == "symptoms must be a JSON list string."


def test_predict_image_message_language(monkeypatch):
    class Model:
        def __call__(self, batch, training=False):
            a = np.zeros((1, 12))
            a[0, 1] = 1
            return a

    monkeypatch.setattr(ip, "load_artifacts", lambda: (Model(), ip.read_labels()))
    assert ip.predict_image(photo())["message"].startswith("Image analyzed.")
    assert ip.predict_image(photo(), "vi")["message"].startswith("Đã phân tích ảnh.")
