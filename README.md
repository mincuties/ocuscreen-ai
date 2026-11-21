---
title: OcuScreen AI
emoji: 👁️
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
---

# OcuScreen AI

A two-stage preliminary screening web app: Keras image model → JSON rule set combined with symptoms → examination priority. This is a prototype that has not been clinically validated; it is not a diagnostic device.

The backend, API messages and documentation are in English. The web interface is bilingual: English by default, with a switch to Vietnamese.

## 1. Repository layout

| Path | Role |
|---|---|
| `model/keras_model.h5` | Trained image model (EfficientNetV2-S, about 198 MB); not in the repo, download it from Drive as described below |
| `model/labels.txt` | All 12 classes in the order verified against the Drive copy |
| `backend/image_predictor.py` | Loads the model once, preprocesses images, runs CPU inference |
| `backend/risk_engine.py` | Combines image signs and symptoms using the JSON rules |
| `backend/i18n.py` | English/Vietnamese message catalog for API responses |
| `backend/app.py` | FastAPI app and static frontend |
| `backend/__init__.py` | Python package marker |
| `data/disease_knowledge.json` | 11 eye labels (including Normal), 28 checklist items, 9 symptom patterns; every user-facing text has `en` and `vi` versions |
| `frontend/index.html` | Semantic page shell: progress, preview, checklist, result |
| `frontend/css/styles.css` | Responsive styles and interaction states |
| `frontend/js/app.js` | Coordinates upload, checklist, language and submit state |
| `frontend/js/i18n.js` | UI strings for English and Vietnamese, language preference |
| `frontend/js/upload.js` | File input, drag and drop, preview URL lifecycle |
| `frontend/js/api.js` | API calls, timeouts and error handling |
| `frontend/js/components.js` | Reusable checklist and result components |
| `frontend/js/validation.js` | File checks and warnings from the knowledge base |
| `backend/request_validation.py` | Upload limits and symptom validation |
| `tests/frontend.test.js` | JavaScript tests on Node, no external libraries |
| `requirements-dev.txt` | Python test dependencies |
| `requirements.txt` | Pinned runtime dependencies |
| `Dockerfile` | Python 3.11, CPU, UID 1000, port 7860 |
| `main.py` | Optional CLI, no `input()` |
| `tests/test_app.py` | Tests for rules, images and the API |

Do not commit `.venv311`, user photos, datasets or secrets. The H5 file is about 198 MB, above GitHub's file size limit, so it is not in the repo. Download it from Drive: https://drive.google.com/file/d/10mt0wgyArY4zrOc53fuQjLfW3ykJNFyI/view . Download it in a browser, keep the name `keras_model.h5` and put it in `model/`. The model loads with TensorFlow 2.15.1 (input `(None,224,224,3)`, softmax output `(None,12)`); prediction quality still needs to be evaluated on real labeled eye photos.

## 2. Defensive design decisions

- Labels are read with `line.strip().split(' ', 1)`, separating the integer ID from the name. The full 0–11 order is checked to avoid mapping outputs to the wrong class.
- Preprocessing matches the original main.py: EXIF orientation → RGB → center resize/crop with `ImageOps.fit(..., (224,224), LANCZOS)` → float32 → `(img / 127.5) - 1.0`.
- The current contract is multiclass softmax `(None,12)` with argmax. Having 12 classes does not mean multi-label. If the actual model uses sigmoid or logits, the API reports a contract error; confirm the output before changing the algorithm.
- Class 11 always asks for a retake and is never used to infer disease. A model score below 0.60 is treated as uncertain. This threshold is a prototype assumption that needs calibration.
- `Normal` does not cancel symptoms. An invalid image with no symptoms returns `Not assessed`, not Low. Symptom warnings still appear when the model cannot load.
- Stage 2 is not trained from the JSON. It reads the rules and ranks symptom patterns with 2 points for a matching image sign + 1 point per symptom; points are never turned into a disease percentage. The High/Medium/Low level is a rule-based examination priority, independent of the pattern ranking.
- One worker, one model, one upload/inference at a time; other requests get 429. This is a resource limit for a small demo, not a complete DDoS defense.
- Limits: 5 MB per image, 12 megapixels, JPEG/PNG/WebP only, no animated images. Paths are based on module location, not the working directory.
- `tensorflow-cpu` avoids CUDA dependencies; the process can still run out of memory on a very small host. Docker uses TF 2.15.1 / Keras 2.15.0, the Keras version the H5 file was saved with.
- The app creates no records and does not write photos or checklists to a database. The multipart parser may use a temporary file and then close it; the hosting platform may keep its own access logs. Only the frontend folder is public through the static route.

## 3. Run locally with Docker (Windows Docker Desktop or WSL)

Put the H5 file in place first, then open a terminal in the folder that contains the Dockerfile:

```bash
docker build -t ocuscreen-ai .
docker run --rm -p 7860:7860 ocuscreen-ai
```

Open http://localhost:7860 . http://localhost:7860/api/health must return `{"ready":true}`. To use port 8000:

```bash
docker run --rm -e PORT=8000 -p 8000:8000 ocuscreen-ai
```

Run directly with Python 3.11 (Linux/WSL):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --host 0.0.0.0 --port 7860 --workers 1
```

On Windows PowerShell use `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1`; the pip and uvicorn commands are the same. Linux Docker is the reference deployment environment.

## 4. Deploying to Hugging Face Spaces step by step

**Cost conditions checked on 2026-09-11:** the current documentation says Docker/Gradio Spaces that run compute need a paid plan to create; CPU Basic has no hourly fee. Docker Spaces cannot be promised to be fully free on a Free account. Check what your account shows before creating one. Source: https://huggingface.co/docs/hub/spaces-overview .

1. Unpack the project and put the original H5 at `model/keras_model.h5`. Do not rename the labels file.
2. Sign in at https://huggingface.co and open https://huggingface.co/new-space .
3. Choose the name `ocuscreen-ai`, SDK **Docker**, template **Blank** if offered. Choose **Public** so outside users can access it. Choose CPU Basic if your account allows it.
4. Note that Public exposes the code and the model in the repo, not only the web page. Only upload assets the team is allowed to share; do not upload other people's test photos.
5. Create the Space. In the **Files** tab, upload the `backend`, `data`, `frontend` and `model` folders plus `requirements.txt`, `Dockerfile` and `README.md` at the repo root. Keep the folder structure. Do not upload a ZIP as the app: HF does not unpack ZIP files to build.
6. Keep the YAML header of this README, including `sdk: docker` and `app_port: 7860`. Commit the change. The Space builds and runs automatically.
7. If uploading folders in the UI is inconvenient, use the CLI on your machine. Replace `YOUR_USERNAME` with your account and run it in the unpacked repo folder:

```bash
python -m pip install huggingface_hub
hf auth login
hf upload YOUR_USERNAME/ocuscreen-ai . . --repo-type space
```

8. Open the **Logs** tab. Wait until the build finishes and the status is **Running**. If the H5 is missing or incompatible, the page still loads but `/api/health` returns 503; do not treat that as a successful deployment.
9. In the **App** tab, use the option to open the app directly and copy the real `.hf.space` URL. You can also share the Space page URL from the address bar. Do not guess the subdomain.
10. Open the link in a private window and test upload, checklist and result. Try `Normal` with reduced vision: the High warning must remain; try a background photo: it must ask for a retake. Confirm labels with real test photos; do not assume the model is always right.

Docker Spaces require the app to listen on `0.0.0.0` and the port declared in the README; this is already configured. Source: https://huggingface.co/docs/hub/spaces-sdks-docker . CLI: https://huggingface.co/docs/huggingface_hub/guides/cli .

## 5. Render option

1. Create a GitHub repo and upload the unpacked structure together with the H5. A private repo works if Render is granted read access.
2. In the Render Dashboard choose **New → Web Service** and connect the repo.
3. Choose the **Docker** runtime, Dockerfile path `./Dockerfile`, with the root at the folder that contains it.
4. Pick a suitable instance. Free has limited resources; TensorFlow may not fit in RAM. On OOM, add RAM or move to a lighter runtime/model; do not add workers.
5. Set `PORT=7860` if needed. The Dockerfile also accepts the PORT value the platform provides. Set the Health Check Path to `/api/health`.
6. Deploy, check the logs and health status, then copy the public URL Render provides.
7. Render Free may sleep after 15 minutes without requests and needs to start up again on the next visit. Do not use Free to promise an always-available service. Source: https://render.com/docs/free .

## 6. API contract

`/api/symptoms` and `/api/screen` accept an optional `lang` query parameter, `en` (default) or `vi`. Unsupported values fall back to English. It selects the language of symptom labels, reasons, recommendations, pattern names and error messages; IDs and model class names stay the same in both languages.

- `GET /api/symptoms?lang=en`: symptom IDs and labels, urgent symptoms and combinations, disclaimer.
- `POST /api/screen?lang=en`: multipart with `image` (file) and `symptoms` (JSON string list of IDs).
- `GET /api/health`: 200 if the model is ready, 503 if it could not load.
- `/`: frontend; `/docs`: OpenAPI.

```bash
curl -X POST 'http://localhost:7860/api/screen?lang=vi' -F 'image=@eye.jpg' -F 'symptoms=["itch","gritty"]'
```

The response contains `image` (class_id, label, confidence, status, message) and `risk` (level, label, urgent, reasons, recommendation, candidates, image_used, knowledge_version, disclaimer). `risk.level` is a language-independent code, `high`, `medium`, `low` or `null` (not assessed); `risk.label` is its translated text. An unreadable image or Back Ground returns 200 with status invalid, plus symptom warnings if any. A model failure returns 503 with the symptom-based risk. Bad input returns 422; an oversized request 413; busy 429. `confidence` is only the model's output score.

## 7. Testing and troubleshooting

Run from the project root, in a Python 3.11 environment with `requirements.txt` installed:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm test
```

`npm test` needs Node 18+ but no `npm install`. The Python tests use a fake model, so the H5 file is not required. The frontend uses ES modules served by FastAPI; do not open the HTML file directly.

- **FileNotFoundError / health 503:** check that `model/keras_model.h5` has the right name and contains real bytes, not a shortcut or LFS pointer.
- **Unrecognized keyword / DepthwiseConv2D groups:** check that you are using the Docker dependencies; take the TensorFlow/Keras version from the original export environment. Do not edit the H5 config JSON or blindly remove layer attributes. Re-export with the correct environment and compare outputs on a fixed image before replacing the model.
- **Wrong shape / output:** the artifact must have input `(None,224,224,3)` and softmax output `(None,12)`. Do not reinterpret sigmoid as softmax.
- **OOM / exit 137:** keep `--workers 1`, do not use reload in the cloud, reduce image size or add host RAM. The CPU package does not guarantee no OOM.
- **API 404:** keep the frontend mount last in app.py and upload all folders.
- **CORS:** the frontend calls relative paths on the same domain; no wildcard CORS is needed.
- **Phone photo rejected:** export as JPEG/PNG/WebP below 12 megapixels; HEIC is not supported yet.

## 8. Knowledge scope

The references in the JSON include NHS pages on red eye, jaundice, bulging eyes and anaemia. These sources support the general descriptions of signs; they do not validate the app's weights or performance. Expert review of the rules, independent test data, calibration and false-negative assessment are needed before any use beyond a demo. An external eye photo cannot detect glaucoma, retinal disease or the cause of cloudiness in the eye. Do not infer a specific disease from the Normal/Cornea Abnormal label alone.

When adding or editing knowledge-base text, provide both `en` and `vi` versions; the test suite checks that every text pair is complete.
