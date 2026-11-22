---
title: OcuScreen AI
emoji: 👁️
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
---

# OcuScreen AI

A two-stage preliminary screening web app: ONNX image model that reports visible eye signs → JSON rule set combined with symptoms → examination priority. This is a prototype that has not been clinically validated; it is not a diagnostic device.

The backend, API messages and documentation are in English. The web interface is bilingual: English by default, with a switch to Vietnamese.

## 1. Repository layout

| Path | Role |
|---|---|
| `model/model_card.json` | Model contract: input preprocessing, output names, per-label thresholds and which labels are published; written by `training/export.py` |
| `model/ocuscreen.onnx` | Image model (frozen DINOv2 ViT-S/14 backbone + linear head); not in the repo, built by `training/` |
| `model/preprocess_golden.npz` | Fixed photos with the tensor and outputs the exporter got, used by a test to catch preprocessing drift |
| `training/` | Feature extraction, head fitting with grouped cross-validation, and ONNX export |
| `backend/image_predictor.py` | Checks the model card, loads the ONNX model once, preprocesses images, runs CPU inference and turns scores into findings |
| `backend/attention.py` | Optional map of which image patches pushed a label's score up |
| `backend/risk_engine.py` | Combines image signs and symptoms using the JSON rules |
| `backend/i18n.py` | English/Vietnamese message catalog for API responses |
| `backend/app.py` | FastAPI app and static frontend |
| `backend/__init__.py` | Python package marker |
| `data/disease_knowledge.json` | 12 image signs (including Normal), 29 checklist items, triage rules, 9 symptom patterns; every user-facing text has `en` and `vi` versions |
| `frontend/index.html` | Semantic page shell: progress, preview, checklist, result |
| `frontend/css/styles.css` | Responsive styles and interaction states |
| `frontend/js/app.js` | Coordinates upload, checklist, language and submit state |
| `frontend/js/i18n.js` | UI strings for English and Vietnamese, language preference |
| `frontend/js/upload.js` | File input, drag and drop, preview URL lifecycle |
| `frontend/js/crop.js` | Square crop box with a circle guide, placed by the user over the photo |
| `frontend/js/resize.js` | Crops and downscales the photo in the browser before upload |
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
| `tests/conftest.py` | Builds a tiny fake ONNX model and card for the tests |
| `tests/triage_cases.json` | Example triage cases, run as tests; to review with an eye specialist |

Do not commit virtual environments, user photos, datasets or secrets. The model files (`ocuscreen.onnx`, `preprocess_golden.npz`) are not in the repo; build them with the steps in section 7 ("Training the model") and put them in `model/` next to `model_card.json`. Prediction quality is only as good as the report `training/fit.py` writes; read it before trusting any label.

## 2. Defensive design decisions

- The server trusts nothing it has not checked: at startup it validates `model_card.json` (format, input spec, label names, which must be image signs in the knowledge base, thresholds), checks the SHA256 of the ONNX file against the card, and checks the session's input and output shapes. Any mismatch keeps the model unloaded and `/api/health` at 503.
- Preprocessing is written in the card and shared with training: EXIF orientation → RGB → `center_crop` (or `letterbox`) to 224 → ImageNet mean/std → NCHW float32. `training/extract.py` calls the same `to_batch` function as the server, and `preprocess_golden.npz` lets a test compare the two on fixed photos. Because a centre crop can cut off an eye near the edge of a wide photo, the page asks the user to place a square over the eye (with a circle as a guide) and sends only that square. MediaPipe face landmarks were not used: they need the whole face and fail on close-ups and everted eyelids, which are the photos this app asks for.
- The model is multi-label: one independent probability per sign, so a photo can show redness and discharge at once. Each label has its own threshold chosen on cross-validation (about 90% sensitivity for signs). Probabilities between the hint threshold and the threshold are `borderline`. A separate `eye_gate` output rejects photos that do not look like an eye.
- A label is only reported if it passed the publish checks in `training/fit.py` (enough independent images, positives from at least two sources, and a high enough AUROC when one source is held out). Signs the model does not report are listed in `unassessed`, and the page says the photo was not checked for them, instead of implying they are absent. With no published Normal label, a photo without any sign is `uncertain`, not normal.
- Publishing a label and letting it change the triage are separate decisions. A published label is shown to the user; it can raise the level on its own (for example, discharge with contact lenses → eye emergency) only when the card marks it `triage: true`, which `training/fit.py` sets only when the label passes the held-out-source check on at least three sources. Otherwise the reasons say the photo may show the sign and ask the user to tick the matching symptom if they notice it. Sign thresholds are also raised until 95% of Normal photos stay below them, and a label is dropped if, together with the others, it flags more than 10% of Normal photos.
- `training/fit.py --demo` shows every trained label even when it fails the checks (the report still lists the failures) and marks the card with `"demo": true`; none of them may raise the triage level, and the page tells the user the model is a demo. The committed card is a demo card: all eight trained signs and Normal are shown, so expect wrong findings on new photos.
- The page shows sign names, never the probabilities: they are calibrated to the training data, where signs are far more common than in real use. The square the user places cannot be smaller than half the photo's short side, because the model is clearly worse on tighter close-ups.
- `Normal` does not cancel symptoms. An invalid image with no symptoms returns `Not assessed`, not Green. Symptom warnings still appear when the model cannot load.
- Stage 2 is not trained from the JSON. It reads the rules and ranks symptom patterns with 2 points for a matching image sign + 1 point per symptom; points are never turned into a disease percentage.
- Triage is a separate step driven by `triage_rules` in the JSON. Each rule has a level (`red` or `yellow`), a destination, and conditions: `all` / `any` symptom IDs, `image` signs (an uncertain image counts only when `hint_ok` is set), `age` and `duration`. The highest matching level wins, and the most urgent destination among the rules at that level is used. With no matching rule the result is `green` (self-care) only if the age group is known and the user confirms they reviewed the whole checklist, because an unchecked box is not evidence that a symptom is absent; otherwise it stays `null`. Rules are checked for typos at startup. They have not been reviewed by an eye specialist yet; `tests/triage_cases.json` lists example cases to review with one.
- One worker, one model, one upload/inference at a time; other requests get 429. This is a resource limit for a small demo, not a complete DDoS defense.
- Limits: 5 MB and 25 megapixels per uploaded image (the page downscales larger photos first), JPEG/PNG/WebP only, no animated images. Paths are based on module location, not the working directory.
- `onnxruntime` (CPU) replaces TensorFlow, which makes the image and its memory use much smaller. `ORT_NUM_THREADS` sets the number of inference threads (default 2 in Docker).
- The app creates no records and does not write photos or checklists to a database. The multipart parser may use a temporary file and then close it; the hosting platform may keep its own access logs. Only the frontend folder is public through the static route.

## 3. Run locally with Docker (Windows Docker Desktop or WSL)

Put `ocuscreen.onnx` and `model_card.json` in `model/` first, then open a terminal in the folder that contains the Dockerfile. The build stops with an error if either is missing, and the image reports its health through `/api/health`:

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

1. Unpack the project and put `ocuscreen.onnx` in `model/` next to `model_card.json`. Do not rename either file.
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

8. Open the **Logs** tab. Wait until the build finishes and the status is **Running**. If the model is missing or does not match its card, the page still loads but `/api/health` returns 503; do not treat that as a successful deployment.
9. In the **App** tab, use the option to open the app directly and copy the real `.hf.space` URL. You can also share the Space page URL from the address bar. Do not guess the subdomain.
10. Open the link in a private window and test upload, checklist and result. Try a photo with no visible sign plus reduced vision: the warning must remain; try a photo that is not an eye: it must ask for a retake. Confirm labels with real test photos; do not assume the model is always right.

Docker Spaces require the app to listen on `0.0.0.0` and the port declared in the README; this is already configured. Source: https://huggingface.co/docs/hub/spaces-sdks-docker . CLI: https://huggingface.co/docs/huggingface_hub/guides/cli .

## 5. Render option

1. Render builds from a Git repository. `model/ocuscreen.onnx` is ignored by `.gitignore`, so it is not pushed with the code and the Docker build fails without it. Use a host that can store the file (for example Hugging Face Spaces, section 4), or add the file to the deployment some other way.
2. In the Render Dashboard choose **New → Web Service** and connect the repo.
3. Choose the **Docker** runtime, Dockerfile path `./Dockerfile`, with the root at the folder that contains it.
4. Pick a suitable instance and check the memory use in the logs after the first requests. The old TensorFlow build used about 657 MB, above the 512 MB of Render Free; the ONNX build has not been measured on Render yet. On OOM, add RAM; do not add workers.
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

The response contains `image` (status, label, confidence, findings, unassessed, unassessed_note, hint, top3, message, attention, model_version, demo (true when the card is a demo card), quality; `class_id` is always `null` and kept for older clients) and `risk` (level, label, urgent, destination, destination_text, reasons, recommendation, first_aid, warnings, rules, candidates, image_used, knowledge_version, disclaimer). `risk.level` is a language-independent code, `red`, `yellow`, `green` or `null` (not assessed); `risk.label` is its translated text and `urgent` is true for red. `destination` is a code (`eye_emergency`, `emergency`, `eye_today`, `doctor_today`, `eye_doctor`, `doctor`, `self_care`); `rules` lists the IDs of every matching rule, or `needs_confirmation` when a green result was withheld. `warnings` carries the corticosteroid eye-drop warning for every assessed result.

`image.status` is `valid` (at least one published sign is above its threshold, or a published Normal is), `uncertain` (nothing is) or `invalid` (unreadable, or the eye gate says it is not an eye photo). An invalid image returns 200 with symptom warnings if any. A model failure returns 503 with the symptom-based risk. Bad input returns 422; an oversized request 413; busy 429. `findings` lists every published sign that is `positive` or `borderline`, with its calibrated probability, threshold and `triage` flag; every positive finding with `triage: true` is passed to the rules, so two signs can both raise the level. `label` and `confidence` repeat the strongest positive finding. `unassessed` lists the image signs this model does not report, and `unassessed_note` says so in the chosen language. `hint` is set only for an uncertain image with one clearly leading borderline sign; it only counts in rules marked `hint_ok`, and only for a `triage` label, and the reasons then say the image result is uncertain. `top3` lists the three highest published scores; the triage never reads it. `model_version` is the SHA256 of the ONNX file. `quality` holds the sharpness (variance of the Laplacian), mean brightness and glare share of the crop the model sees, the flags they raise (`blurry`, `dark`, `bright`, `glare`) and translated retake tips in `advice`; it is `null` when the image could not be read. The flags never change the image status or the triage, and their thresholds in `image_predictor.py` are uncalibrated. With `explain=true`, `attention` holds a map for the headline sign (a JPEG data URL of the 224×224 crop the model saw, coloured on the 16×16 patches whose features pushed that sign's score up) and a note, only for a valid label other than Normal; otherwise it is `null`. The head is linear, so this map is exact for the model (no gradients needed), but it shows what drove the score, not where a lesion is, and the triage never reads it.

## 7. Testing and troubleshooting

Run from the project root, in a Python 3.11 environment with `requirements.txt` installed:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
npm test
```

`npm test` needs Node 18+ but no `npm install`. The Python tests build a tiny fake ONNX model, so the real model is not required; the golden preprocessing test is skipped until `model/preprocess_golden.npz` exists. The frontend uses ES modules served by FastAPI; do not open the HTML file directly.

- **Health 503:** the log says why. Usually `ocuscreen.onnx` is missing, or its SHA256 does not match `model_card.json` because the two files come from different exports. Copy both from the same `training/export.py` run.
- **Card errors:** a label name that is not an image sign in `data/disease_knowledge.json`, or a threshold outside 0–1, stops the model from loading. Fix the training output, not the check.
- **OOM / exit 137:** keep `--workers 1`, do not use reload in the cloud, lower `ORT_NUM_THREADS` or add host RAM.
- **API 404:** keep the frontend mount last in app.py and upload all folders.
- **CORS:** the frontend calls relative paths on the same domain; no wildcard CORS is needed.
- **Phone photo rejected:** the web page downscales photos to a 1600 px long edge before upload; direct API calls must send JPEG/PNG/WebP up to 5 MB and 25 megapixels. HEIC is not supported yet.

### Training the model

Training needs Python 3.10–3.12, PyTorch and the packages in `training/requirements.txt`, plus internet access the first time to download the DINOv2 weights through `torch.hub`. A GPU helps but is not required. The dataset folder must contain `labels.csv` with the columns `file`, `split` (`train`, `val`, `test`), `dup_group`, `source`, `normal` and one 0/1 column per sign (see `SIGNS` in `training/common.py`).

```bash
python -m pip install -r training/requirements.txt
python training/extract.py --data "path/to/dataset"   # embeds 16 augmented views per photo
python training/fit.py --data "path/to/dataset"       # cross-validation, thresholds, report
python training/export.py --data "path/to/dataset"    # writes model/ocuscreen.onnx and the card
```

The backbone stays frozen: `extract.py` caches its features once (in `training/out/`), so `fit.py` only fits a logistic-regression head and runs in minutes. Augmented views use random crops, flips, blur, JPEG and small brightness changes, never colour shifts, because colour is the sign for redness, yellow eye and pallor. `fit.py` chooses regularisation, calibration and thresholds on repeated cross-validation grouped by `dup_group`, so near-duplicate photos never sit on both sides of a split, and scores the test split only once at the end. Read `training/out/report.md`: it says which labels were published and why the others were not. `export.py` folds the head into the ONNX graph, checks that onnxruntime matches PyTorch on test photos, and writes the card and the golden file.

### Evaluating the model

`scripts/evaluate.py` measures an exported model on labeled photos it was not trained on, with the same preprocessing as the server. Pass the dataset's `labels.csv` (it uses the `test` split by default), or a CSV with `path,labels` columns where `labels` is a `;`-separated list of sign names:

```bash
python scripts/evaluate.py "path/to/dataset/labels.csv" --crop 1.0,0.7,0.5 --out report.json
```

For each label it prints AUROC, average precision, and sensitivity, specificity (95% Wilson intervals) and PPV at the card's threshold, plus calibration error. `--crop` repeats the run on centre crops of each photo, to check that close-ups (like the square the page sends) score as well as wide photos.

### After retraining the model

- Compare `center_crop` and `letterbox` in the card on the same test set with `scripts/evaluate.py`, and keep the better one (extract again with the same setting).
- Recalibrate the hint margin and the photo-quality flags from the evaluation report.
- Check the attention maps on real photos: they should sit on the eye, not on the cheek or the background.
- Have an eye specialist review the rules and the cases in `tests/triage_cases.json`.

## 8. Knowledge scope

The references in the JSON include NHS pages on red eye, jaundice, bulging eyes and anaemia. These sources support the general descriptions of signs; they do not validate the app's weights or performance. Expert review of the rules, independent test data, calibration and false-negative assessment are needed before any use beyond a demo. An external eye photo cannot detect glaucoma, retinal disease or the cause of cloudiness in the eye. Do not infer a specific disease from one image sign alone.

When adding or editing knowledge-base text, provide both `en` and `vi` versions; the test suite checks that every text pair is complete.

## 9. License

The code is released under the MIT License; see `LICENSE`. The license covers this repository's code, not the model weights or third-party datasets.
