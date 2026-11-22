import io
import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import backend.app as api
import backend.image_predictor as ip
from pathlib import Path
import backend.risk_engine as engine
from backend.risk_engine import assess_risk


def prediction(label="Normal", status="valid"):
    return {
        "label": label,
        "status": status,
        "class_id": None,
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
    monkeypatch.setattr(api, "predict_image", lambda b, lang="en", explain=False: prediction())
    with TestClient(api.app) as c:
        yield c


CASES = json.loads(
    (Path(__file__).parent / "triage_cases.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_triage_cases(case):
    options = {k: case[k] for k in ("age_group", "duration", "confirmed") if k in case}
    risk = assess_risk(case["image"], case["symptoms"], **options)
    assert (risk["level"], risk["destination"]) == (case["level"], case["destination"])
    assert risk["urgent"] == (case["level"] == "red")


MILD = [s["id"] for s in api.KB["symptoms"] if s.get("mild")]
CONTEXT = [s["id"] for s in api.KB["symptoms"] if s.get("context")]


@pytest.mark.parametrize("sid", [s["id"] for s in api.KB["symptoms"]])
def test_every_symptom_alone(sid):
    for age in ("child", "adult"):
        for duration in (None, "lt_1d", "1_7d"):
            risk = assess_risk(
                prediction(), [sid], age_group=age, duration=duration, confirmed=True
            )
            # Mild symptoms and context alone are green; everything else needs a doctor.
            expected = {"green"} if sid in MILD + CONTEXT else {"yellow", "red"}
            assert risk["level"] in expected, (sid, age, duration)
        risk = assess_risk(
            prediction(), [sid], age_group=age, duration="gt_7d", confirmed=True
        )
        expected = {"green"} if sid in CONTEXT else {"yellow"} if sid in MILD else {
            "yellow",
            "red",
        }
        assert risk["level"] in expected, (sid, age, "gt_7d")
    # Unknown age never gives green; babies are never sent to self-care.
    unknown = assess_risk(prediction(), [sid], confirmed=True)
    assert unknown["level"] != "green"
    infant = assess_risk(prediction(), [sid], age_group="infant", confirmed=True)
    assert infant["level"] == ("yellow" if sid in CONTEXT else "red")


def test_green_needs_confirmation_and_normal_never_lowers():
    risk = assess_risk(prediction(), ["itch"])
    assert risk["level"] is None and risk["rules"] == ["needs_confirmation"]
    assert "confirm" in risk["recommendation"]
    assert assess_risk(prediction(), ["vision_loss"])["level"] == "red"
    assert assess_risk(prediction(), ["pain"])["level"] == "yellow"


def test_first_aid_and_steroid_warning():
    chemical = assess_risk(prediction(), ["chemical", "contacts", "red_eye"])
    assert [a.split()[0] for a in chemical["first_aid"]] == ["Rinse", "Take"]
    assert chemical["rules"][:2] == ["chemical", "contacts_symptoms"]
    assert chemical["destination_text"].startswith("Eye emergency")
    assert "corticosteroids" in chemical["warnings"][0]
    green = assess_risk(prediction(), [], age_group="adult", confirmed=True)
    assert green["warnings"] and not green["first_aid"]
    unknown = assess_risk(prediction(None, "invalid"), [], confirmed=True)
    assert unknown["level"] is None and unknown["warnings"] == []


def test_reasons_name_the_symptoms_and_signs():
    risk = assess_risk(prediction(), ["photophobia", "vision_loss"])
    assert risk["reasons"] == [
        "Danger signs: Sudden reduced or lost vision, Pain when looking at light"
    ]
    risk = assess_risk(prediction("Redness"), ["contacts"], "vi")
    assert risk["reasons"][0] == "Đang dùng kính áp tròng; ảnh gợi ý: Đỏ mắt"
    mild = assess_risk(
        prediction(None, "invalid"), ["itch"], age_group="adult", confirmed=True
    )
    assert mild["reasons"][0] == "Only mild symptoms: Itchy eyes"
    assert "could not be used" in mild["reasons"][-1]


def test_unknown_triage_values_are_rejected():
    with pytest.raises(ValueError):
        assess_risk(prediction(), [], age_group="teen")
    with pytest.raises(ValueError):
        assess_risk(prediction(), [], duration="forever")


@pytest.mark.parametrize(
    "change",
    [
        {"all": ["chemicals"]},  # unknown symptom
        {"image": ["Red"]},  # unknown image sign
        {"level": "orange"},
        {"destination": "pharmacy"},
        {"duration": ["1d"]},
        {"any": []},
        {"imgae": ["Redness"]},  # unknown key
        {"reason": {"en": "x", "vi": " "}},
        {"reason": "x"},
        {"reason": {"en": "{foo}", "vi": "x"}},  # unknown placeholder
        {"all": []},
        {"hint_ok": True},  # without an image condition
        {"image": ["Redness"], "hint_ok": "yes"},
    ],
)
def test_rule_typos_fail_at_startup(change):
    rule = {**api.KB["triage_rules"][0], **change}
    with pytest.raises(ValueError):
        engine.check_rules([rule])


def test_duplicate_rule_ids_fail_at_startup():
    rule = api.KB["triage_rules"][0]
    with pytest.raises(ValueError):
        engine.check_rules([rule, rule])


@pytest.mark.parametrize(
    "probs,hint",
    [
        ({"Swelling": 0.45}, "Swelling"),
        ({"Swelling": 0.25}, None),  # below the hint threshold
        ({"Swelling": 0.45, "Redness": 0.40}, None),  # no clear lead
        ({"Swelling": 0.45, "Redness": 0.30}, "Swelling"),
        ({"Swelling": 0.70}, None),  # valid, not a hint
        ({"Cataract": 0.45}, None),  # no hint threshold for this label
        ({"Swelling": 0.45, "eye_gate": 0.1}, None),  # not an eye photo
    ],
)
def test_uncertain_hint(fake_model, probs, hint):
    fake_model(probs)
    result = ip.predict_image(photo())
    assert (result["hint"] or {}).get("label") == hint


def hinted(label):
    pred = prediction(label, "uncertain")
    pred["hint"] = {"label": label, "confidence": 0.45}
    return pred


def test_hint_counts_only_where_a_rule_allows_it():
    risk = assess_risk(hinted("Cornea Abnormal"), [])
    assert risk["level"] == "yellow" and not risk["urgent"]
    assert risk["image_used"] is False and risk["candidates"] == []
    assert "uncertain" in risk["reasons"][-1]
    # A corneal hint with pain is not enough for red; a valid corneal sign is.
    assert assess_risk(hinted("Cornea Abnormal"), ["pain"])["level"] == "yellow"
    assert assess_risk(prediction("Cornea Abnormal"), ["pain"])["level"] == "red"
    # Medium-priority hints alone are ignored.
    assert assess_risk(hinted("Redness"), [], confirmed=True)["level"] is None
    mild = assess_risk(hinted("Redness"), ["itch"], age_group="adult", confirmed=True)
    assert mild["level"] == "green" and "could not be used" in mild["reasons"][-1]
    assert assess_risk(hinted("Redness"), ["contacts"])["level"] == "red"
    assert assess_risk(prediction("Redness", "uncertain"), ["contacts"])["level"] is None


def test_hint_never_lowers_an_urgent_result():
    risk = assess_risk(hinted("Cornea Abnormal"), ["vision_loss"])
    assert risk["level"] == "red" and risk["urgent"]


def test_file_sha256(tmp_path):
    f = tmp_path / "m.h5"
    f.write_bytes(b"abc")
    assert ip.file_sha256(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_card_is_checked_against_the_onnx_file(fake_model):
    fake_model()
    session, card = ip.load_artifacts()
    assert [label["name"] for label in card["labels"]][-1] == "eye_gate"
    assert ip.model_version == card["sha256"]


def _bad_sha(card):
    card["sha256"] = "0" * 64


def _unknown_label(card):
    card["labels"][0]["name"] = "Pink Eye"


def _no_threshold(card):
    del card["labels"][1]["threshold"]


def _hint_above_threshold(card):
    card["labels"][0]["hint_threshold"] = 0.9


def _published_gate(card):
    card["labels"][-1]["published"] = True


def _wrong_size(card):
    card["input"]["size"] = 256


def _bad_resize(card):
    card["input"]["resize"] = "stretch"


def _missing_label(card):
    card["labels"].pop(0)
    for i, label in enumerate(card["labels"]):
        label["index"] = i


@pytest.mark.parametrize(
    "edit",
    [_bad_sha, _unknown_label, _no_threshold, _hint_above_threshold, _published_gate,
     _wrong_size, _bad_resize, _missing_label],
)
def test_bad_model_card_refuses_to_load(fake_model, edit):
    fake_model(card_edit=edit)
    with pytest.raises(ip.ModelUnavailable):
        ip.load_artifacts()


def test_missing_model_files_refuse_to_load(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ip, "_session", None)
    with pytest.raises(ip.ModelUnavailable):
        ip.load_artifacts()


def test_preprocess():
    a = ip.preprocess(photo())
    assert a.shape == (1, 3, 224, 224) and a.dtype == np.float32
    mean, std = np.array(ip.DEFAULT_INPUT["mean"]), np.array(ip.DEFAULT_INPUT["std"])
    # Channels first: every pixel of channel c is (1 - mean[c]) / std[c] for white.
    assert np.allclose(a[0, :, 0, 0], (1 - mean) / std, atol=1e-6)
    assert np.allclose(a[0], ((1 - mean) / std)[:, None, None], atol=1e-6)
    assert np.allclose(ip.preprocess(photo("black"))[0, :, 5, 5], -mean / std, atol=1e-6)
    red = io.BytesIO()
    Image.new("RGB", (300, 240), (255, 0, 0)).save(red, format="PNG")
    channels = ip.preprocess(red.getvalue())[0, :, 0, 0]
    assert channels[0] > 0 > channels[1] and channels[2] < 0
    with pytest.raises(ip.InvalidImage):
        ip.preprocess(b"not an image")


def test_letterbox_keeps_the_whole_photo():
    image = Image.new("RGB", (400, 200), "white")
    spec = {**ip.DEFAULT_INPUT, "resize": "letterbox"}
    a = ip.to_batch(image, spec)
    mean, std = np.array(spec["mean"]), np.array(spec["std"])
    assert a.shape == (1, 3, 224, 224)
    # The padding (top row) is the mean colour, i.e. close to 0 after normalizing.
    assert np.allclose(a[0, :, 0, 112], 0, atol=0.02)
    assert np.allclose(a[0, :, 112, 112], (1 - mean) / std, atol=1e-5)


def test_real_model_matches_the_training_preprocessing():
    """Only with a real model in model/: same tensor and logits as at export time."""
    golden = ip.MODEL_DIR / "preprocess_golden.npz"
    if not golden.is_file() or not (ip.MODEL_DIR / ip.CARD_NAME).is_file():
        pytest.skip("no real model in model/")
    data = np.load(golden)
    session, card = ip.load_artifacts()
    batch = ip.to_batch(ip.decode(data["image_bytes"].tobytes()), card["input"])
    assert np.allclose(batch, data["tensor"], atol=1e-5)
    logits, _ = ip.run_model(session, card, batch)
    assert np.allclose(logits, data["logits"][0], atol=1e-3)


def jpeg(width, height):
    b = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(b, format="JPEG", quality=70)
    return b.getvalue()


def test_preprocess_accepts_default_phone_photo():
    # 4032x3024 is the default 12 MP photo of many phones (12,192,768 pixels).
    assert ip.preprocess(jpeg(4032, 3024)).shape == (1, 3, 224, 224)


def invalid_key(data):
    with pytest.raises(ip.InvalidImage) as err:
        ip.preprocess(data)
    return err.value.key


@pytest.mark.parametrize("size", [(1500, 1000), (1500, 1500)])
def test_preprocess_too_many_pixels_says_so(monkeypatch, size):
    # A lowered limit keeps the test light: 1.5 MP is above it (Pillow warning),
    # 2.25 MP is above twice it (Pillow error). Both must give the size message.
    monkeypatch.setattr(ip, "MAX_PIXELS", 1_000_000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    assert invalid_key(jpeg(*size)) == "image_dimensions"


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_preprocess_rejects_other_formats(fmt):
    b = io.BytesIO()
    Image.new("RGB", (300, 240), "white").save(b, format=fmt)
    with pytest.raises(ip.InvalidImage):
        ip.preprocess(b.getvalue())


def textured(blur=0):
    from PIL import ImageFilter

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.normal(128, 40, (480, 640, 3)).clip(0, 255).astype("uint8"))
    b = io.BytesIO()
    img.filter(ImageFilter.GaussianBlur(blur)).save(b, format="PNG")
    return b.getvalue()


@pytest.mark.parametrize(
    "data,flags",
    [
        (textured(), []),
        (textured(blur=6), ["blurry"]),
        (photo("black"), ["blurry", "dark"]),
        (photo("white"), ["blurry", "bright", "glare"]),
    ],
)
def test_quality_flags(data, flags):
    assert ip.measure_quality(ip.decode(data))["flags"] == flags


def test_glare_spot_is_flagged_only_when_large():
    rng = np.random.default_rng(1)
    pixels = rng.normal(110, 30, (448, 448, 3)).clip(0, 240).astype("uint8")
    small, large = pixels.copy(), pixels.copy()
    small[200:210, 200:210] = 255  # a catchlight, about 0.05% of the crop
    large[100:200, 100:200] = 255  # about 5% of the crop

    def flags(a):
        return ip.measure_quality(Image.fromarray(a))["flags"]
    assert flags(small) == [] and flags(large) == ["glare"]


def test_quality_flags_never_change_the_result(fake_model):
    fake_model({"Redness": 0.9})
    result = ip.predict_image(photo("black"), "vi")
    assert result["status"] == "valid" and result["label"] == "Redness"
    assert result["quality"]["flags"] == ["blurry", "dark"]
    assert result["quality"]["advice"][1].startswith("Ảnh có vẻ quá tối")


def test_top3_is_ordered_and_does_not_affect_triage(fake_model):
    fake_model({"Redness": 0.62, "Discharge": 0.2, "Swelling": 0.1, "Yellow Eye": 0.9})
    result = ip.predict_image(photo())
    # Unpublished labels (Yellow Eye) are never shown.
    assert [t["label"] for t in result["top3"]] == ["Redness", "Discharge", "Swelling"]
    assert result["top3"][1]["confidence"] == pytest.approx(0.2, abs=1e-4)
    with_top3 = assess_risk(result, [], age_group="adult")
    without = assess_risk({**result, "top3": None}, [], age_group="adult")
    assert with_top3 == without


def test_several_signs_are_found_and_all_reach_the_triage(fake_model):
    fake_model({"Redness": 0.8, "Discharge": 0.6, "Swelling": 0.35})
    result = ip.predict_image(photo())
    assert result["status"] == "valid" and result["label"] == "Redness"
    assert result["confidence"] == pytest.approx(0.8, abs=1e-4)
    states = {f["label"]: f["state"] for f in result["findings"]}
    assert states == {"Redness": "positive", "Discharge": "positive", "Swelling": "borderline"}
    risk = assess_risk(result, [], age_group="adult")
    assert risk["level"] == "yellow" and risk["image_used"]
    assert "Eye redness" in risk["reasons"][0] and "discharge" in risk["reasons"][0].lower()
    # A borderline sign of a valid photo is not used by the triage.
    assert "swelling" not in risk["reasons"][0].lower()


def test_unassessed_signs_are_listed_and_block_a_normal_reading(fake_model):
    fake_model({"Normal": 0.95})
    result = ip.predict_image(photo())
    assert result["status"] == "valid" and result["label"] == "Normal"
    assert "Yellow Eye" in result["unassessed"] and "Cornea Abnormal" in result["unassessed"]
    assert "Redness" not in result["unassessed"]
    assert "Yellowish eye" in result["unassessed_note"] or "yellow" in result["unassessed_note"].lower()


def test_unpublished_normal_never_reads_as_normal(fake_model):
    def hide_normal(card):
        next(l for l in card["labels"] if l["name"] == "Normal")["published"] = False

    fake_model({"Normal": 0.99}, card_edit=hide_normal)
    result = ip.predict_image(photo())
    assert result["status"] == "uncertain" and result["label"] is None
    assert "Normal" not in result["unassessed"]


def test_low_normal_and_no_sign_is_uncertain(fake_model):
    fake_model({"Normal": 0.5})
    result = ip.predict_image(photo())
    assert result["status"] == "uncertain" and result["findings"] == []


def test_predict_not_an_eye(fake_model):
    fake_model({"eye_gate": 0.05, "Redness": 0.99})
    result = ip.predict_image(photo())
    assert result["status"] == "invalid" and result["label"] is None
    assert result["findings"] == [] and result["top3"] is None
    assert assess_risk(result, [], confirmed=True)["image_used"] is False


def test_routes(client):
    assert client.get("/").status_code == 200
    assert len(client.get("/api/symptoms").json()["symptoms"]) == 29
    assert client.get("/model/model_card.json").status_code == 404
    r = client.post(
        "/api/screen",
        files={"image": ("eye.png", photo(), "image/png")},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.status_code == 200 and r.json()["risk"]["level"] == "red"


def screen(client, **data):
    return client.post(
        "/api/screen", files={"image": ("eye.png", photo(), "image/png")}, data=data
    )


def test_screen_triage_fields(client):
    assert screen(client, symptoms='["itch"]').json()["risk"]["level"] is None
    risk = screen(client, symptoms='["itch"]', confirmed="true").json()["risk"]
    assert risk["level"] is None  # age group missing
    risk = screen(
        client, symptoms='["itch"]', confirmed="true", age_group="adult"
    ).json()["risk"]
    assert risk["level"] == "green" and risk["destination"] == "self_care"
    risk = screen(client, symptoms='["itch"]', age_group="infant").json()["risk"]
    assert risk["level"] == "red" and risk["rules"][0] == "infant_symptoms"
    risk = screen(client, symptoms='["blurred"]', duration="gt_7d").json()["risk"]
    assert risk["level"] == "yellow"
    # Empty values from an unselected <select> mean "not given".
    risk = screen(client, symptoms='["blurred"]', duration="", age_group="").json()
    assert risk["risk"]["level"] == "red"


@pytest.mark.parametrize(
    "data",
    [{"age_group": "teen"}, {"duration": "2d"}, {"confirmed": "yes"}, {"explain": "1"}],
)
def test_screen_rejects_bad_triage_fields(client, data):
    response = screen(client, **data)
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Invalid age group")
    assert not api.app.state.busy


def test_screen_passes_the_explain_choice(client, monkeypatch):
    calls = []

    def fake(b, lang="en", explain=False):
        calls.append(explain)
        return prediction()

    monkeypatch.setattr(api, "predict_image", fake)
    screen(client)
    screen(client, explain="true")
    assert calls == [False, True]


def test_screen_rejects_duplicate_triage_fields(client):
    response = client.post("/api/screen", files=[
        ("image", ("eye.png", photo(), "image/png")),
        ("age_group", (None, "adult")),
        ("age_group", (None, "infant")),
    ])
    assert response.status_code == 422


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
    def invalid(b, lang="en", explain=False):
        raise ip.InvalidImage("image_unreadable")

    monkeypatch.setattr(api, "predict_image", invalid)
    r = client.post(
        "/api/screen",
        files={"image": ("e.png", b"bad")},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.json()["risk"]["level"] == "red"
    assert r.json()["image"]["status"] == "invalid"
    assert r.json()["image"]["quality"] is None
    assert r.json()["image"]["top3"] is None


def test_missing_model_preserves_warning(client, monkeypatch):
    def missing(b, lang="en", explain=False):
        raise ip.ModelUnavailable("missing")

    monkeypatch.setattr(api, "predict_image", missing)
    r = client.post(
        "/api/screen",
        files={"image": ("e.png", photo())},
        data={"symptoms": '["vision_loss"]'},
    )
    assert r.status_code == 503 and r.json()["risk"]["level"] == "red"


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


def test_inference_failure_is_model_unavailable(fake_model, monkeypatch):
    fake_model()
    session, card = ip.load_artifacts()

    class Broken:
        def run(self, *args, **kwargs):
            raise RuntimeError("device unavailable")

    monkeypatch.setattr(ip, "load_artifacts", lambda: (Broken(), card))
    with pytest.raises(ip.ModelUnavailable):
        ip.predict_image(photo())


def test_non_finite_output_is_model_unavailable(fake_model, monkeypatch):
    fake_model()
    session, card = ip.load_artifacts()

    class NaN:
        def run(self, *args, **kwargs):
            n = len(card["labels"])
            return np.full((1, n), np.nan), np.zeros((1, 256, n))

    monkeypatch.setattr(ip, "load_artifacts", lambda: (NaN(), card))
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
    rules = client.get("/api/symptoms").json()["red_rules"]
    assert [r["id"] for r in rules] == [
        r["id"]
        for r in api.KB["triage_rules"]
        if r["level"] == "red" and "image" not in r
    ]
    assert "chemical" in [r["id"] for r in rules]
    assert all(isinstance(r["reason"], str) for r in rules)


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
    # 29 symptoms + 12 image signs + 29 rules + 10 patterns x 2 + 4 sources + 2
    assert len(pairs) == 96
    assert all(p["en"].strip() and p["vi"].strip() for p in pairs)


def test_risk_language_defaults_to_english():
    pred = prediction("Yellow Eye")
    en = assess_risk(pred, ["vision_loss"])
    vi = assess_risk(pred, ["vision_loss"], "vi")
    assert en["level"] == vi["level"] == "red"
    assert en["label"] == "Red · go now" and vi["label"] == "Đỏ · đi khám ngay"
    assert "Danger signs: Sudden reduced or lost vision" in en["reasons"]
    assert "Dấu hiệu nguy hiểm: Giảm / mất thị lực đột ngột" in vi["reasons"]
    assert en["candidates"][0]["name"] == "Yellow eyes needing a general check-up"
    assert vi["candidates"][0]["name"] == "Vàng mắt cần kiểm tra toàn thân"
    assert assess_risk(prediction(None, "invalid"), [])["label"] == "Not assessed"


def test_symptoms_endpoint_language(client):
    en = client.get("/api/symptoms").json()
    vi = client.get("/api/symptoms?lang=vi").json()
    fallback = client.get("/api/symptoms?lang=fr").json()
    assert en["lang"] == fallback["lang"] == "en" and vi["lang"] == "vi"
    assert en["symptoms"][1] == {"id": "itch", "label": "Itchy eyes", "mild": True}
    assert vi["symptoms"][1] == {"id": "itch", "label": "Ngứa mắt", "mild": True}
    assert fallback["symptoms"] == en["symptoms"]
    assert en["disclaimer"].startswith("Experimental")


def test_screen_errors_follow_language(client, monkeypatch):
    def invalid(b, lang="en", explain=False):
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


def test_predict_image_message_language(fake_model):
    fake_model({"Redness": 0.9})
    assert ip.predict_image(photo())["message"].startswith("Image analyzed.")
    assert ip.predict_image(photo(), "vi")["message"].startswith("Đã phân tích ảnh.")


def test_cataract_sign_is_never_green():
    risk = assess_risk(prediction("Cataract"), ["itch"], age_group="adult", confirmed=True)
    assert risk["level"] == "yellow" and risk["destination"] == "eye_doctor"


def multi(*labels, triage=True):
    pred = prediction(labels[0])
    pred["findings"] = [
        {"label": label, "probability": 0.8, "threshold": 0.5, "state": "positive", "triage": triage}
        for label in labels
    ]
    return pred


def test_any_positive_sign_can_fire_a_rule():
    # Swelling is listed first, but the contact-lens rule needs Discharge.
    risk = assess_risk(multi("Swelling", "Discharge"), ["contacts"])
    assert risk["level"] == "red" and "contacts_image" in risk["rules"]
    assert risk["candidates"][0]["match_points"] >= 2


def test_normal_with_findings_is_ignored_and_never_lowers():
    pred = multi("Redness")
    pred["label"] = "Normal"
    assert assess_risk(pred, ["contacts"])["level"] == "red"


def test_display_only_sign_never_raises_the_level():
    # Discharge with contact lenses is red when the card trusts it for triage...
    assert assess_risk(multi("Discharge"), ["contacts"])["level"] == "red"
    # ...but a display-only Discharge leaves only the symptom-based result.
    risk = assess_risk(multi("Discharge", triage=False), ["contacts"], age_group="adult", confirmed=True)
    assert risk["level"] == "green" and "contacts_image" not in risk["rules"]
    assert any("Discharge" in r or "discharge" in r.lower() for r in risk["reasons"])
    assert risk["image_used"] is False
    # Infants: a display-only sign must not trigger the emergency rule.
    assert assess_risk(multi("Swelling", triage=False), [], age_group="infant")["level"] != "red"


def test_display_only_sign_alone_still_needs_confirmation():
    risk = assess_risk(multi("Swelling", triage=False), [])
    assert risk["level"] is None and risk["rules"] == ["needs_confirmation"]


def test_display_only_hint_is_not_used():
    pred = prediction("Swelling")
    pred.update(status="uncertain", label=None, findings=[], hint={"label": "Swelling", "confidence": 0.4, "triage": False})
    assert "infant_image" not in assess_risk(pred, [], age_group="infant")["rules"]
    pred["hint"]["triage"] = True
    assert "infant_image" in assess_risk(pred, [], age_group="infant")["rules"]


def test_predictor_marks_display_only_findings(fake_model):
    fake_model({"Watery": 0.8, "Redness": 0.8})
    result = ip.predict_image(photo())
    triage = {f["label"]: f["triage"] for f in result["findings"]}
    assert triage == {"Watery": False, "Redness": True}
    risk = assess_risk(result, [], age_group="adult", confirmed=True)
    assert "image_sign" in risk["rules"]


def test_card_rejects_triage_on_hidden_label(fake_model):
    def bad(card):
        next(l for l in card["labels"] if l["name"] == "Yellow Eye")["triage"] = True

    fake_model(card_edit=bad)
    with pytest.raises(ValueError):
        ip.read_card()


def test_demo_flag_is_passed_through_and_validated(fake_model):
    fake_model({"Redness": 0.8})
    assert ip.predict_image(photo())["demo"] is False

    def demo(card):
        card["demo"] = True

    fake_model({"Redness": 0.8}, card_edit=demo)
    assert ip.predict_image(photo())["demo"] is True

    def bad(card):
        card["demo"] = "yes"

    fake_model(card_edit=bad)
    with pytest.raises(ValueError):
        ip.read_card()
