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
| `backend/attention.py` | Optional Grad-CAM map of where the model looked |
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
| `tests/triage_cases.json` | Example triage cases, run as tests; to review with an eye specialist |

Do not commit `.venv311`, user photos, datasets or secrets. The H5 file is about 198 MB, above GitHub's file size limit, so it is not in the repo. Download it from Drive: https://drive.google.com/file/d/10mt0wgyArY4zrOc53fuQjLfW3ykJNFyI/view . Download it in a browser, keep the name `keras_model.h5` and put it in `model/`. The model loads with TensorFlow 2.15.1 (input `(None,224,224,3)`, softmax output `(None,12)`); prediction quality still needs to be evaluated on real labeled eye photos.

## 2. Defensive design decisions

- Labels are read with `line.strip().split(' ', 1)`, separating the integer ID from the name. The full 0–11 order is checked to avoid mapping outputs to the wrong class.
- Preprocessing matches the original main.py: EXIF orientation → RGB → center resize/crop with `ImageOps.fit(..., (224,224), LANCZOS)` → float32 → `(img / 127.5) - 1.0`.
- The current contract is multiclass softmax `(None,12)` with argmax. Having 12 classes does not mean multi-label. If the actual model uses sigmoid or logits, the API reports a contract error; confirm the output before changing the algorithm.
- Class 11 always asks for a retake and is never used to infer disease. A model score below 0.60 is treated as uncertain. This threshold is a prototype assumption that needs calibration.
- `Normal` does not cancel symptoms. An invalid image with no symptoms returns `Not assessed`, not Green. Symptom warnings still appear when the model cannot load.
- Stage 2 is not trained from the JSON. It reads the rules and ranks symptom patterns with 2 points for a matching image sign + 1 point per symptom; points are never turned into a disease percentage.
- Triage is a separate step driven by `triage_rules` in the JSON. Each rule has a level (`red` or `yellow`), a destination, and conditions: `all` / `any` symptom IDs, `image` signs (an uncertain image counts only when `hint_ok` is set), `age` and `duration`. The highest matching level wins, and the most urgent destination among the rules at that level is used. With no matching rule the result is `green` (self-care) only if the age group is known and the user confirms they reviewed the whole checklist, because an unchecked box is not evidence that a symptom is absent; otherwise it stays `null`. Rules are checked for typos at startup. They have not been reviewed by an eye specialist yet; `tests/triage_cases.json` lists example cases to review with one.
- One worker, one model, one upload/inference at a time; other requests get 429. This is a resource limit for a small demo, not a complete DDoS defense.
- Limits: 5 MB and 25 megapixels per uploaded image (the page downscales larger photos first), JPEG/PNG/WebP only, no animated images. Paths are based on module location, not the working directory.
- `tensorflow-cpu` avoids CUDA dependencies; the process can still run out of memory on a very small host. Docker uses TF 2.15.1 / Keras 2.15.0, the Keras version the H5 file was saved with.
- The app creates no records and does not write photos or checklists to a database. The multipart parser may use a temporary file and then close it; the hosting platform may keep its own access logs. Only the frontend folder is public through the static route.

## 3. Run locally with Docker (Windows Docker Desktop or WSL)

Put the H5 file in place first, then open a terminal in the folder that contains the Dockerfile. The build stops with an error if `model/keras_model.h5` is missing, and the image reports its health through `/api/health`:

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

1. Render builds from a Git repository. The current H5 (about 198 MB) is above GitHub's 100 MB file limit and is ignored by `.gitignore`, so it cannot be pushed with the code; the Docker build fails without it. Use a host that can store the file (for example Hugging Face Spaces, section 4) or a model file below 100 MB.
2. In the Render Dashboard choose **New → Web Service** and connect the repo.
3. Choose the **Docker** runtime, Dockerfile path `./Dockerfile`, with the root at the folder that contains it.
4. Pick a suitable instance. The server was measured at about 657 MB of RAM with the current model, above the 512 MB of Render Free, so Free is expected to run out of memory. On OOM, add RAM or move to a lighter runtime/model; do not add workers.
5. Set `PORT=7860` if needed. The Dockerfile also accepts the PORT value the platform provides. Set the Health Check Path to `/api/health`.
6. Deploy, check the logs and health status, then copy the public URL Render provides.
7. Render Free may sleep after 15 minutes without requests and needs to start up again on the next visit. Do not use Free to promise an always-available service. Source: https://render.com/docs/free .

## 6. API contract

`/api/symptoms` and `/api/screen` accept an optional `lang` query parameter, `en` (default) or `vi`. Unsupported values fall back to English. It selects the language of symptom labels, reasons, recommendations, pattern names and error messages; IDs and model class names stay the same in both languages.

- `GET /api/symptoms?lang=en`: symptom IDs and labels, the red rules that need no image (`red_rules`, so the page can warn before upload), disclaimer.
- `POST /api/screen?lang=en`: multipart with `image` (file), `symptoms` (JSON string list of IDs) and optional `age_group` (`infant` under 3 months, `child`, `adult`), `duration` (`lt_1d`, `1_7d`, `gt_7d`) `confirmed` (`true` or `false`, default `false`) and `explain` (`true` or `false`, default `false`). Empty or missing values mean unknown.
- `GET /api/health`: 200 if the model is ready, 503 if it could not load.
- `/`: frontend; `/docs`: OpenAPI.

```bash
curl -X POST 'http://localhost:7860/api/screen?lang=vi' -F 'image=@eye.jpg' -F 'symptoms=["itch","gritty"]' -F age_group=adult -F duration=1_7d -F confirmed=true
```

The response contains `image` (class_id, label, confidence, status, message, hint, model_version, quality, top3, attention) and `risk` (level, label, urgent, destination, destination_text, reasons, recommendation, first_aid, warnings, rules, candidates, image_used, knowledge_version, disclaimer). `risk.level` is a language-independent code, `red`, `yellow`, `green` or `null` (not assessed); `risk.label` is its translated text and `urgent` is true for red. `destination` is a code (`eye_emergency`, `emergency`, `eye_today`, `doctor_today`, `eye_doctor`, `doctor`, `self_care`); `rules` lists the IDs of every matching rule, or `needs_confirmation` when a green result was withheld. `warnings` carries the corticosteroid eye-drop warning for every assessed result. An unreadable image or Back Ground returns 200 with status invalid, plus symptom warnings if any. A model failure returns 503 with the symptom-based risk. Bad input returns 422; an oversized request 413; busy 429. `confidence` is only the model's output score. `hint` is set only for an uncertain image whose top class clearly leads (uncalibrated guards in `image_predictor.py`); it only counts in rules marked `hint_ok`, and the reasons then say the image result is uncertain. `model_version` is the SHA256 of the loaded H5 file. `quality` holds the sharpness (variance of the Laplacian), mean brightness and glare share of the centre crop the model sees, the flags they raise (`blurry`, `dark`, `bright`, `glare`) and translated retake tips in `advice`; it is `null` when the image could not be read. The flags never change the image status or the triage, and their thresholds in `image_predictor.py` are uncalibrated. `top3` lists the three highest model scores; the triage never reads it, and the page shows the runners-up only for a valid image, without Back Ground or scores under 0.15. The page hides the label of an uncertain image and prints the model and rule versions with the result. With `explain=true`, `attention` holds a Grad-CAM map (a JPEG data URL of the 224×224 crop the model saw, coloured where the model looked) and a note, only for a valid label other than Normal; otherwise it is `null`. It shows what drove the model's score, not where a lesion is, and the triage never reads it. Grad-CAM++ was tried first and dropped: with this model's pooling head it gave nearly the same map for every class (see `backend/attention.py`). Colours are scaled to each photo's own peak. The map needs a model that is a plain chain of layers ending in a softmax Dense layer, as the current one is; for any other model it is turned off at startup with a warning in the log. It must be checked again after the model is retrained.

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
- **Phone photo rejected:** the web page downscales photos to a 1600 px long edge before upload; direct API calls must send JPEG/PNG/WebP up to 5 MB and 25 megapixels. HEIC is not supported yet.
- **Smaller model file:** `python scripts/strip_optimizer.py model/keras_model.h5 model/keras_model_slim.h5` re-saves the model without its optimizer state (about 198 MB → 82 MB, identical outputs, checked by the script). RAM use while serving stays about the same.

### Evaluating the model

`scripts/evaluate.py` measures a model on labeled photos it was not trained on. Put the photos in one folder per class, named exactly as in `model/labels.txt` (for example `test/Cornea Abnormal/001.jpg`), or list them in a CSV with `path,label` columns:

```bash
python scripts/evaluate.py test/ --model model/keras_model.h5 --out report.json
```

It prints, per class, sensitivity and specificity with 95% Wilson intervals, PPV and one-vs-rest AUROC, then the share of images accepted and their accuracy at several top-1 thresholds (to choose the 0.60 threshold from data), and the accuracy on images with each photo-quality flag (to check whether the flags really mark photos the model gets wrong). The JSON adds the confusion matrix and calibration bins (ECE). Images go through the same preprocessing as the server. Keep photos of the same person in one set, and run both model files on the same set to compare them.

## 8. Knowledge scope

The references in the JSON include NHS pages on red eye, jaundice, bulging eyes and anaemia. These sources support the general descriptions of signs; they do not validate the app's weights or performance. Expert review of the rules, independent test data, calibration and false-negative assessment are needed before any use beyond a demo. An external eye photo cannot detect glaucoma, retinal disease or the cause of cloudiness in the eye. Do not infer a specific disease from the Normal/Cornea Abnormal label alone.

When adding or editing knowledge-base text, provide both `en` and `vi` versions; the test suite checks that every text pair is complete.

## 9. License

The code is released under the MIT License; see `LICENSE`. The license covers this repository's code, not the model weights or third-party datasets.
