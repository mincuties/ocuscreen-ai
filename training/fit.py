"""Fit the linear head on cached embeddings, pick thresholds and write the report.

    python training/fit.py --data "C:/path/_clean_v1"

Reads training/out/features.npz (from extract.py) and writes training/out/head.npz,
training/out/report.json and training/out/report.md.

The test split is only scored once, at the end, with the head refitted on train+val.
Everything else (regularisation, calibration, thresholds, which labels are published)
comes from repeated grouped cross-validation on train+val.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from common import GATE, NORMAL, OUT, SIGNS, output_names, read_rows, trained_columns
from backend import metrics

C_GRID = (0.01, 0.03, 0.1, 0.3)
SENSITIVITY_TARGET = 0.90
NORMAL_SPECIFICITY = 0.95  # share of images with a sign that must stay below the Normal threshold
HINT_SPECIFICITY = 0.90
# Share of Normal photos that must stay below each sign's threshold (raises the threshold
# above the sensitivity target when needed; the report shows the sensitivity that is left).
SIGN_NORMAL_SPECIFICITY = 0.95
# Eye gate: reject at least this share of real non-eye photos (OOF) while passing at
# least GATE_EYE_MIN_PASS of eye photos; otherwise fall back to GATE_EYE_MIN_PASS.
GATE_NON_EYE_REJECT = 0.95
GATE_EYE_MIN_PASS = 0.95
GATE_C = 0.1
# Publication criteria; a label that misses one stays in the head but is hidden.
MIN_GROUPS = 30
MIN_SOURCE_POSITIVES = 5
MIN_SOURCES = 2
MIN_LOSO_AUROC = 0.80
MIN_SPECIFICITY = 0.80
MAX_NORMAL_ANY_SIGN = 0.10
# A published label is shown to the user; it may also raise the triage level on its own
# only with positives from at least this many sources that each pass the LOSO check.
TRIAGE_MIN_SOURCES = 3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    return mu, sd


def fit_many(Z, Y, Cs, row_weight):
    """Independent L2-regularised logistic regressions, one per column of Y, fitted together.

    Minimises, for each label j, C_j * sum_i row_weight_i * cw_j(y_ij) * logloss_ij
    + ||w_j||^2 / 2 (intercept not penalised), the same objective as scikit-learn's
    LogisticRegression(class_weight="balanced"). ``row_weight`` lets the views of one
    image share a weight of 1, so more views do not mean weaker regularisation.
    Returns (W of shape (d, L), b of shape (L,)); a label with one class gets w = 0 and a
    saturated intercept.
    """
    from scipy.optimize import minimize

    Z = np.asarray(Z, np.float64)
    Y = np.asarray(Y, np.float64)
    n, d = Z.shape
    L = Y.shape[1]
    Cs = np.asarray(Cs, np.float64)
    pos = (Y * row_weight[:, None]).sum(0)
    neg = ((1 - Y) * row_weight[:, None]).sum(0)
    total = row_weight.sum()
    ok = (pos > 0) & (neg > 0)
    cw_pos = np.where(ok, total / (2 * np.maximum(pos, 1e-12)), 0)
    cw_neg = np.where(ok, total / (2 * np.maximum(neg, 1e-12)), 0)
    S = row_weight[:, None] * np.where(Y == 1, cw_pos, cw_neg) * Cs  # (n, L)

    def objective(theta):
        W = theta[: d * L].reshape(d, L)
        b = theta[d * L:]
        A = Z @ W + b
        # log(1 + exp(-y*a)) with y in {-1, 1}, computed stably
        sgn = 2 * Y - 1
        m = -sgn * A
        loss = np.logaddexp(0, m)
        g = -sgn * (1 / (1 + np.exp(-m)))  # d loss / d a
        f = (S * loss).sum() + 0.5 * (W * W).sum()
        G = S * g
        gW = Z.T @ G + W
        gb = G.sum(0)
        return f, np.concatenate([gW.ravel(), gb])

    theta0 = np.zeros(d * L + L)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    W = result.x[: d * L].reshape(d, L)
    b = result.x[d * L:]
    for j in np.where(~ok)[0]:
        W[:, j] = 0
        b[j] = 20.0 if pos[j] > 0 else -20.0
    return W, b


def view_weight(img):
    """1 / (number of rows of the same image), so every image weighs 1 in the loss."""
    _, inverse, counts = np.unique(img, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


class Features:
    def __init__(self, path, n_images):
        data = np.load(path)
        meta = Path(path).with_suffix(".json")
        self.negatives_real = (
            json.loads(meta.read_text(encoding="utf-8")).get("negatives_real") if meta.exists() else None
        )
        self.X = data["X"].astype(np.float32)
        self.kind = data["kind"]
        self.index = data["index"]
        self.view = data["view"]
        eye = self.kind == "eye"
        self.view0 = np.full(n_images, -1)
        sel = np.where(eye & (self.view == 0))[0]
        self.view0[self.index[sel]] = sel
        self.n_views = int(self.view[eye].max()) + 1

    def rows(self, images, max_view):
        """Rows of all views < max_view of the given images, with the image of each row."""
        mask = (self.kind == "eye") & (self.view < max_view) & np.isin(self.index, images)
        idx = np.where(mask)[0]
        return idx, self.index[idx]


def threshold_for_sensitivity(probs, y, target):
    """Highest threshold that keeps sensitivity >= target."""
    pos = np.sort(probs[y == 1])
    if not len(pos):
        return math.nan
    k = int(math.floor((1 - target) * len(pos)))
    return float(pos[k])


def threshold_for_specificity(probs, y, target):
    """Lowest threshold with specificity >= target (negatives strictly below it)."""
    neg = np.sort(probs[y == 0])
    if not len(neg):
        return math.nan
    k = min(len(neg) - 1, int(math.ceil(target * len(neg))) - 1)
    return float(np.nextafter(neg[k], 1))


def rarest_key(Y, counts):
    """Stratification key: the rarest label each image has (or -1)."""
    order = np.argsort(counts)
    key = np.full(len(Y), -1)
    for j in order[::-1]:
        key[Y[:, j] == 1] = j
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="folder with labels.csv")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-views", type=int, default=8, help="views per image used inside CV")
    args = parser.parse_args()

    rows = read_rows(args.data)
    columns = trained_columns(rows)
    names = output_names(columns)
    F = Features(OUT / "features.npz", len(rows))
    if len(F.view0) != len(rows) or (F.view0 < 0).any():
        missing = int((F.view0 < 0).sum())
        print(f"warning: {missing} images have no embedding and are left out")
    ok = F.view0 >= 0
    Y = np.array([[r[c] for c in columns] for r in rows], dtype=int)
    groups = np.array([r["dup_group"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    split = np.array([r["split"] for r in rows])
    fitset = np.where((split != "test") & ok)[0]
    testset = np.where((split == "test") & ok)[0]
    cv_views = min(args.cv_views, F.n_views)
    L = len(columns)
    counts = Y[fitset].sum(axis=0)
    key = rarest_key(Y[fitset], counts)

    # ---- repeated grouped CV: out-of-fold logits per repeat (view 0 only) ----
    oof = {}  # (repeat, C) -> (len(fitset), L)
    best_c = None
    for r in range(args.repeats):
        cgrid = C_GRID if r == 0 else None
        splitter = StratifiedGroupKFold(args.folds, shuffle=True, random_state=r)
        for fold, (tr, va) in enumerate(splitter.split(fitset, key, groups[fitset])):
            tr_img, va_img = fitset[tr], fitset[va]
            idx, img = F.rows(tr_img, cv_views)
            mu, sd = standardize(F.X[idx])
            Ztr = (F.X[idx] - mu) / sd
            Zva = (F.X[F.view0[va_img]] - mu) / sd
            weight = view_weight(img)
            for C in cgrid or [None]:
                Cs = [C] * L if C is not None else best_c
                Wf, bf = fit_many(Ztr, Y[img], Cs, weight)
                out = oof.setdefault((r, C), np.full((len(fitset), L), np.nan))
                out[va] = Zva @ Wf + bf
            print(f"repeat {r} fold {fold} done", flush=True)
        if r == 0:
            best_c = []
            for j in range(L):
                ap = {C: metrics.average_precision(oof[(0, C)][:, j], Y[fitset, j]) for C in C_GRID}
                best_c.append(max(ap, key=lambda c: (np.nan_to_num(ap[c], nan=-1), -c)))
    O = np.stack([
        np.array([oof[(0, best_c[j])][:, j] for j in range(L)]).T
    ] + [oof[(r, None)] for r in range(1, args.repeats)])  # (repeats, n_fit, L)

    # ---- calibration and thresholds ----
    Yf = Y[fitset]
    normal_col = columns.index(NORMAL)
    platt, thresholds, hints, label_reports = [], [], [], []
    for j, column in enumerate(columns):
        y = Yf[:, j]
        x = O[:, :, j].reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(x, np.tile(y, args.repeats))
        a, b = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        platt.append([a, b])
        P = sigmoid(a * O[:, :, j] + b)
        if column == NORMAL:
            t = max(threshold_for_specificity(P[r], y, NORMAL_SPECIFICITY) for r in range(args.repeats))
            hint = None
        else:
            t_sens = min(threshold_for_sensitivity(P[r], y, SENSITIVITY_TARGET) for r in range(args.repeats))
            normal_only = (y == 0) & (Yf[:, normal_col] == 1)
            t_norm = max(
                threshold_for_specificity(P[r][normal_only | (y == 1)], y[normal_only | (y == 1)], SIGN_NORMAL_SPECIFICITY)
                for r in range(args.repeats)
            ) if normal_only.any() else -math.inf
            t = max(t_sens, t_norm)
            th = threshold_for_specificity(P.reshape(-1), np.tile(y, args.repeats), HINT_SPECIFICITY)
            hint = th if th < t else None
        thresholds.append(t)
        hints.append(hint)
        pos = y == 1
        src = Counter(sources[fitset][pos])
        label_reports.append({
            "label": names[j],
            "column": column,
            "C": best_c[j],
            "positives": int(pos.sum()),
            "groups": int(len(set(groups[fitset][pos]))),
            "sources": dict(src),
            "auroc_oof": float(np.nanmean([metrics.auroc(P[r], y) for r in range(args.repeats)])),
            "ap_oof": float(np.nanmean([metrics.average_precision(P[r], y) for r in range(args.repeats)])),
            "oof_at_threshold": metrics.binary_rates(P.reshape(-1), np.tile(y, args.repeats), t),
            "threshold": t,
            "hint_threshold": hint,
            "platt": [a, b],
        })

    # ---- leave-one-source-out: positives from an unseen source vs held-out negatives ----
    rng = np.random.default_rng(0)
    for j, rep in enumerate(label_reports):
        eligible = [s for s, n in rep["sources"].items() if n >= MIN_SOURCE_POSITIVES]
        rep["loso_auroc"] = {}
        if len(eligible) < MIN_SOURCES:
            continue
        for s in eligible:
            other_groups = np.unique(groups[fitset][sources[fitset] != s])
            held_groups = set(rng.choice(other_groups, size=len(other_groups) // 4, replace=False))
            held = np.array([g in held_groups for g in groups[fitset]])
            in_s = sources[fitset] == s
            train_img = fitset[~in_s & ~held]
            idx, img = F.rows(train_img, cv_views)
            mu, sd = standardize(F.X[idx])
            Wl, bl = fit_many((F.X[idx] - mu) / sd, Y[img][:, [j]], [best_c[j]], view_weight(img))
            test_img = fitset[in_s | held]
            scores = ((F.X[F.view0[test_img]] - mu) / sd) @ Wl[:, 0] + bl[0]
            y = Y[test_img, j]
            from_s = sources[test_img] == s
            keep = (from_s & (y == 1)) | (y == 0)
            rep["loso_auroc"][s] = metrics.auroc(scores[keep], y[keep])

    # ---- source probe: how well the embedding tells the source apart ----
    probe_correct = []
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    Xv0 = F.X[F.view0[fitset]]
    for tr, va in splitter.split(fitset, sources[fitset], groups[fitset]):
        mu, sd = standardize(Xv0[tr])
        clf = LogisticRegression(C=0.1, max_iter=3000).fit((Xv0[tr] - mu) / sd, sources[fitset][tr])
        probe_correct.append(clf.predict((Xv0[va] - mu) / sd) == sources[fitset][va])
    probe_acc = float(np.concatenate(probe_correct).mean())
    majority = Counter(sources[fitset]).most_common(1)[0][1] / len(fitset)

    # ---- publication decision ----
    for j, rep in enumerate(label_reports):
        loso = [v for v in rep["loso_auroc"].values() if not math.isnan(v)]
        checks = {
            "groups": rep["groups"] >= MIN_GROUPS,
            "sources": len(loso) >= MIN_SOURCES,
            "loso_auroc": bool(loso) and min(loso) >= MIN_LOSO_AUROC,
            "specificity": rep["oof_at_threshold"]["specificity"] >= MIN_SPECIFICITY,
        }
        rep["checks"] = checks
        rep["published"] = all(checks.values())
        rep["triage"] = rep["published"] and sum(v >= MIN_LOSO_AUROC for v in loso) >= TRIAGE_MIN_SOURCES

    # Share of Normal images that get at least one published sign (false-alarm load), over
    # every repeat. If it is too high, drop the published sign with the lowest specificity.
    is_normal = Yf[:, normal_col] == 1

    def normal_any_rate(js):
        if not js or not is_normal.any():
            return None
        rates = []
        for r in range(args.repeats):
            P0 = np.stack([sigmoid(platt[j][0] * O[r, :, j] + platt[j][1]) for j in js], 1)
            rates.append((P0 >= np.array([thresholds[j] for j in js])).any(1)[is_normal].mean())
        return float(np.mean(rates))

    sign_pub = [j for j, c in enumerate(columns) if c != NORMAL and label_reports[j]["published"]]
    any_rate = normal_any_rate(sign_pub)
    while any_rate is not None and any_rate > MAX_NORMAL_ANY_SIGN:
        worst = min(sign_pub, key=lambda j: label_reports[j]["oof_at_threshold"]["specificity"])
        label_reports[worst]["checks"]["normal_any_sign"] = False
        label_reports[worst]["published"] = label_reports[worst]["triage"] = False
        sign_pub.remove(worst)
        any_rate = normal_any_rate(sign_pub)

    # ---- eye gate ----
    neg_rows = np.where(F.kind == "neg")[0]
    neg_img = F.index[neg_rows]
    eye_img_all = np.where(ok)[0]
    gate_oof_eye, gate_oof_neg, gate_neg_ids = [], [], []
    kf_eye = np.array_split(np.random.default_rng(1).permutation(eye_img_all), 5)
    neg_ids = np.unique(neg_img)
    real_neg = int(F.negatives_real) if F.negatives_real is not None else len(neg_ids)
    kf_neg = np.array_split(np.random.default_rng(2).permutation(neg_ids), 5)
    for k in range(5):
        tr_eye = np.setdiff1d(eye_img_all, kf_eye[k])
        idx, eimg = F.rows(tr_eye, cv_views)
        keep = ~np.isin(neg_img, kf_neg[k])
        nrow = neg_rows[keep]
        Xg = np.concatenate([F.X[idx], F.X[nrow]])
        yg = np.r_[np.ones(len(idx)), np.zeros(len(nrow))]
        wg = np.r_[view_weight(eimg), view_weight(neg_img[keep])]
        mu, sd = standardize(Xg)
        Wg, bg = fit_many((Xg - mu) / sd, yg[:, None], [GATE_C], wg)
        gate_oof_eye.append(((F.X[F.view0[kf_eye[k]]] - mu) / sd) @ Wg[:, 0] + bg[0])
        held_neg = neg_rows[np.isin(neg_img, kf_neg[k]) & (F.view[neg_rows] == 0)]
        gate_oof_neg.append(((F.X[held_neg] - mu) / sd) @ Wg[:, 0] + bg[0])
        gate_neg_ids.append(F.index[held_neg])
    ge, gn = sigmoid(np.concatenate(gate_oof_eye)), sigmoid(np.concatenate(gate_oof_neg))
    is_real = np.concatenate(gate_neg_ids) < real_neg
    gn_real = gn[is_real]
    # Lowest threshold that rejects enough real non-eye photos, unless that fails too many
    # eye photos; then the threshold that passes exactly GATE_EYE_MIN_PASS of them.
    t_reject = threshold_for_specificity(gn_real, np.zeros(len(gn_real)), GATE_NON_EYE_REJECT) if len(gn_real) else -math.inf
    t_eye = float(np.quantile(ge, 1 - GATE_EYE_MIN_PASS))
    gate_t = float(min(t_reject, t_eye))
    gate_report = {
        "threshold": gate_t,
        "eye_pass_oof": float((ge >= gate_t).mean()),
        "real_non_eye_rejected_oof": float((gn_real < gate_t).mean()) if len(gn_real) else None,
        "synthetic_rejected_oof": float((gn[~is_real] < gate_t).mean()) if (~is_real).any() else None,
        "real_non_eye_images": real_neg,
        "synthetic_images": int(len(neg_ids) - real_neg),
    }

    # ---- final head on all train+val views, folded into raw feature space ----
    idx, img = F.rows(fitset, F.n_views)
    mu, sd = standardize(F.X[idx])
    Z = (F.X[idx] - mu) / sd
    W = np.zeros((F.X.shape[1], L + 1))
    bias = np.zeros(L + 1)
    Wf, bf = fit_many(Z, Y[img], best_c, view_weight(img))
    W[:, :L] = Wf / sd[:, None]
    bias[:L] = bf - (Wf * (mu / sd)[:, None]).sum(axis=0)
    gidx, gimg = F.rows(eye_img_all, F.n_views)
    Xg = np.concatenate([F.X[gidx], F.X[neg_rows]])
    yg = np.r_[np.ones(len(gidx)), np.zeros(len(neg_rows))]
    wg = np.r_[view_weight(gimg), view_weight(neg_img)]
    gmu, gsd = standardize(Xg)
    Wg, bg = fit_many((Xg - gmu) / gsd, yg[:, None], [GATE_C], wg)
    W[:, L] = Wg[:, 0] / gsd
    bias[L] = bg[0] - float((Wg[:, 0] * gmu / gsd).sum())
    half = F.X.shape[1] // 2
    patch_mean = F.X[idx][:, half:].mean(axis=0)

    # ---- one look at the test split ----
    Xt = F.X[F.view0[testset]]
    raw = Xt @ W + bias
    test_reports = []
    for j, rep in enumerate(label_reports):
        p = sigmoid(platt[j][0] * raw[:, j] + platt[j][1])
        y = Y[testset, j]
        entry = {
            "label": rep["label"],
            **metrics.binary_rates(p, y, thresholds[j]),
            "auroc": metrics.auroc(p, y),
            "ap": metrics.average_precision(p, y),
            "ece": metrics.binary_ece(p, y),
        }
        zoom_rows = np.where(F.kind == "zoom")[0]
        for z, name in enumerate(("zoom70", "zoom50")):
            sel = zoom_rows[F.view[zoom_rows] == z]
            pos = {i: k for k, i in enumerate(testset)}
            keep = [k for k, i in enumerate(F.index[sel]) if i in pos]
            if not keep:
                continue
            ti = np.array([pos[i] for i in F.index[sel][keep]])
            pz = sigmoid(platt[j][0] * (F.X[sel][keep] @ W[:, j] + bias[j]) + platt[j][1])
            entry[name] = {"auroc": metrics.auroc(pz, y[ti]), **metrics.binary_rates(pz, y[ti], thresholds[j])}
        test_reports.append(entry)
    gate_test = float((sigmoid(raw[:, L]) >= gate_t).mean())
    zoom_rows = np.where(F.kind == "zoom")[0]
    test_ids = set(testset.tolist())
    gate_zoom = {}
    for z, name in enumerate(("zoom70", "zoom50")):
        sel = [i for i in zoom_rows[F.view[zoom_rows] == z] if F.index[i] in test_ids]
        if sel:
            gate_zoom[name] = float((sigmoid(F.X[sel] @ W[:, L] + bias[L]) >= gate_t).mean())

    np.savez(
        OUT / "head.npz",
        W=W.astype(np.float32),
        b=bias.astype(np.float32),
        patch_mean=patch_mean.astype(np.float32),
        names=np.array(names),
    )
    card_labels = []
    for j, rep in enumerate(label_reports):
        card_labels.append({
            "name": rep["label"],
            "index": j,
            "published": rep["published"],
            "triage": rep["triage"],
            "platt": rep["platt"],
            "threshold": rep["threshold"],
            "hint_threshold": rep["hint_threshold"],
        })
    card_labels.append({
        "name": GATE, "index": L, "published": False, "triage": False, "platt": [1.0, 0.0],
        "threshold": gate_t, "hint_threshold": None,
    })
    report = {
        "columns": columns,
        "repeats": args.repeats,
        "folds": args.folds,
        "cv_views": cv_views,
        "views": F.n_views,
        "fit_images": int(len(fitset)),
        "test_images": int(len(testset)),
        "labels": label_reports,
        "test": test_reports,
        "gate": {**gate_report, "eye_pass_test": gate_test, "eye_pass_test_zoom": gate_zoom},
        "source_probe": {"accuracy": probe_acc, "majority_baseline": majority},
        "normal_any_sign_rate_oof": any_rate,
        "card_labels": card_labels,
        "criteria": {
            "sensitivity_target": SENSITIVITY_TARGET,
            "min_groups": MIN_GROUPS,
            "min_sources": MIN_SOURCES,
            "min_source_positives": MIN_SOURCE_POSITIVES,
            "min_loso_auroc": MIN_LOSO_AUROC,
            "min_specificity": MIN_SPECIFICITY,
            "sign_normal_specificity": SIGN_NORMAL_SPECIFICITY,
            "max_normal_any_sign": MAX_NORMAL_ANY_SIGN,
            "triage_min_sources": TRIAGE_MIN_SOURCES,
        },
    }
    (OUT / "report.json").write_text(json.dumps(no_nan(report), indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(summary(report), encoding="utf-8")
    print(summary(report))


def no_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return no_nan(value.item())
    if isinstance(value, dict):
        return {k: no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [no_nan(v) for v in value]
    return value


def fmt(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def summary(report):
    lines = [
        f"Train+val images: {report['fit_images']}, test images: {report['test_images']}, "
        f"{report['repeats']}x{report['folds']}-fold grouped CV, {report['cv_views']} views in CV, "
        f"{report['views']} in the final fit.",
        "",
        "## Cross-validation (train+val, out of fold)",
        "",
        "| Label | Pos | Groups | Sources (>=5 pos) | AUROC | AP | Threshold | Sens | Spec | LOSO AUROC (min) | Published |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["labels"]:
        loso = [v for v in r["loso_auroc"].values() if v is not None and not math.isnan(v)]
        srcs = sum(1 for n in r["sources"].values() if n >= MIN_SOURCE_POSITIVES)
        o = r["oof_at_threshold"]
        failed = [k for k, v in r["checks"].items() if not v]
        lines.append(
            f"| {r['label']} | {r['positives']} | {r['groups']} | {srcs} | {fmt(r['auroc_oof'])} | "
            f"{fmt(r['ap_oof'])} | {fmt(r['threshold'])} | {fmt(o['sensitivity'])} | "
            f"{fmt(o['specificity'])} | {fmt(min(loso)) if loso else '-'} | "
            f"{('yes, triage' if r['triage'] else 'yes, display only') if r['published'] else 'no (' + ', '.join(failed) + ')'} |"
        )
    lines += [
        "",
        "## Test split (scored once)",
        "",
        "| Label | Pos | Sens (95% CI) | Spec (95% CI) | AUROC | AP | ECE | AUROC zoom 70% | AUROC zoom 50% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in report["test"]:
        lo, hi = t["sensitivity_ci"]
        slo, shi = t["specificity_ci"]
        lines.append(
            f"| {t['label']} | {t['n_pos']} | {fmt(t['sensitivity'])} ({fmt(lo)}-{fmt(hi)}) | "
            f"{fmt(t['specificity'])} ({fmt(slo)}-{fmt(shi)}) | {fmt(t['auroc'])} | {fmt(t['ap'])} | "
            f"{fmt(t['ece'])} | {fmt(t.get('zoom70', {}).get('auroc'))} | {fmt(t.get('zoom50', {}).get('auroc'))} |"
        )
    g, p = report["gate"], report["source_probe"]
    lines += [
        "",
        f"Eye gate: threshold {fmt(g['threshold'])}, eye photos passing {fmt(g['eye_pass_oof'])} (OOF) / "
        f"{fmt(g['eye_pass_test'])} (test; zoom 70% {fmt(g['eye_pass_test_zoom'].get('zoom70'))}, "
        f"50% {fmt(g['eye_pass_test_zoom'].get('zoom50'))}), real non-eye photos rejected "
        f"{fmt(g['real_non_eye_rejected_oof'])} ({g['real_non_eye_images']} photos), synthetic "
        f"{fmt(g['synthetic_rejected_oof'])} ({g['synthetic_images']}).",
        f"Source probe: a linear classifier names the photo source with accuracy {fmt(p['accuracy'])} "
        f"(majority baseline {fmt(p['majority_baseline'])}). High accuracy means labels and sources can be confused.",
        f"Normal photos with at least one published sign (OOF): {fmt(report['normal_any_sign_rate_oof'])}.",
        "",
        "The test photos come from the same sources as the training photos, so these numbers do not "
        "show how the model does on other cameras or people. Look at the test split once; if a "
        "setting changes after that, the test numbers are no longer an unbiased estimate.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
