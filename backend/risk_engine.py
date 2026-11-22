"""Transparent prototype triage rules. No trained disease probabilities."""

import json
from pathlib import Path

from backend.i18n import DEFAULT, localize, t

KB = json.loads(
    (Path(__file__).resolve().parents[1] / "data/disease_knowledge.json").read_text(
        encoding="utf-8"
    )
)
SYMPTOMS = {s["id"]: s for s in KB["symptoms"]}
CONTEXT_SYMPTOMS = frozenset(s["id"] for s in KB["symptoms"] if s.get("context"))
AGE_GROUPS = ("infant", "child", "adult")
DURATIONS = ("lt_1d", "1_7d", "gt_7d")
LEVELS = ("red", "yellow")
DESTINATIONS = (
    "eye_emergency", "emergency", "eye_today", "doctor_today", "eye_doctor", "doctor"
)
FIRST_AID = ("rinse", "injury_care", "remove_lenses")
RULE_KEYS = frozenset(
    "id level destination all any image hint_ok age duration first_aid reason".split()
)
DISEASES = [(disease, frozenset(disease["symptoms"])) for disease in KB["diseases"]]


def check_rules(rules):
    """Fail at startup on a typo that would silently disable a safety rule."""
    signs = KB["image_signs"].keys()
    ids = set()
    for rule in rules:
        rid = rule.get("id")
        problems = [
            rid in ids,
            rule.get("level") not in LEVELS,
            rule.get("destination") not in DESTINATIONS,
            not set(rule.get("all", [])) <= SYMPTOMS.keys(),
            "all" in rule and not rule["all"],
            not set(rule.get("any", [])) <= SYMPTOMS.keys(),
            "any" in rule and not rule["any"],
            not set(rule.get("image", [])) <= signs,
            "image" in rule and not rule["image"],
            not set(rule.get("age", [])) <= set(AGE_GROUPS),
            not set(rule.get("duration", [])) <= {*DURATIONS, "unknown"},
            not set(rule.get("first_aid", [])) <= set(FIRST_AID),
            not {"all", "any", "image", "age"} & rule.keys(),
            not rule.keys() <= RULE_KEYS,
            rule.get("hint_ok", False) not in (True, False),
            "hint_ok" in rule and "image" not in rule,
            not reason_ok(rule.get("reason")),
        ]
        if any(problems):
            raise ValueError(f"Invalid triage rule: {rid!r}")
        ids.add(rid)
    return rules


def reason_ok(reason):
    if not isinstance(reason, dict) or set(reason) != {"en", "vi"}:
        return False
    try:
        texts = [str(text).format(symptoms="x", sign="x") for text in reason.values()]
    except (KeyError, IndexError, ValueError):
        return False
    return all(text.strip() for text in texts)


RULES = check_rules(KB["triage_rules"])


def localized_symptoms(lang=DEFAULT):
    return [{**s, "label": localize(s["label"], lang)} for s in KB["symptoms"]]


def pre_upload_rules(lang=DEFAULT):
    """Red rules that need no image, so the page can warn before the upload."""
    return [
        {**rule, "reason": localize(rule["reason"], lang)}
        for rule in RULES
        if rule["level"] == "red" and "image" not in rule
    ]


def image_signs(prediction):
    """Signs the image supports: every positive finding of a valid photo that the model
    card allows into the triage (``triage``). Other positive findings are shown to the user
    but never change the level; see display_only_signs.

    A prediction without ``findings`` (a single-label result) counts its label.
    Normal only stands alone; it never removes a sign and is never matched by a rule.
    """
    if prediction.get("status") != "valid":
        return set()
    findings = prediction.get("findings")
    if findings is None:
        found = {prediction.get("label")}
    else:
        positive = [f for f in findings if f.get("state") == "positive"]
        found = {f["label"] for f in positive if f.get("triage")}
        if not positive and prediction.get("label") == "Normal":
            found = {"Normal"}
    return found & KB["image_signs"].keys()


def display_only_signs(prediction):
    """Positive findings of a valid photo that the triage does not use, in model order."""
    if prediction.get("status") != "valid":
        return []
    return [
        f["label"]
        for f in prediction.get("findings") or []
        if f.get("state") == "positive" and not f.get("triage") and f["label"] in KB["image_signs"]
    ]


def match(rule, selected, age_group, duration, signs, hint_label):
    """Return (matched symptoms, matched image signs, via hint) or None."""
    if not set(rule.get("all", [])) <= selected:
        return None
    if "any" in rule and not selected & set(rule["any"]):
        return None
    if "age" in rule and age_group not in rule["age"]:
        return None
    if "duration" in rule and duration not in rule["duration"]:
        return None
    images, via_hint = [], False
    if "image" in rule:
        images = [s for s in rule["image"] if s in signs]
        if not images and rule.get("hint_ok") and hint_label in rule["image"]:
            images, via_hint = [hint_label], True
        elif not images:
            return None
    # Name the symptoms that made an "any" rule fire; "all" ones are in its reason.
    named = selected & set(rule.get("any") or rule.get("all", []))
    return sorted(named, key=list(SYMPTOMS).index), images, via_hint


def assess_risk(
    prediction,
    symptom_ids,
    lang=DEFAULT,
    *,
    age_group=None,
    duration=None,
    confirmed=False,
):
    selected = set(symptom_ids)
    if selected - SYMPTOMS.keys():
        raise ValueError("Unknown symptom ID.")
    if age_group not in (None, *AGE_GROUPS) or duration not in (None, *DURATIONS):
        raise ValueError("Unknown age group or duration.")
    age_group = age_group or "unknown"
    duration = duration or "unknown"

    def label(sid):
        return localize(SYMPTOMS[sid]["label"], lang)

    def describe(sign):
        return localize(KB["image_signs"][sign]["description"], lang)

    signs = image_signs(prediction)
    # An uncertain image can only count where a rule allows it (hint_ok).
    hint = prediction.get("hint") or {}
    hint_label = hint.get("label") if not signs and hint.get("triage", True) else None
    matched = []
    for rule in RULES:
        found = match(rule, selected, age_group, duration, signs, hint_label)
        if found:
            matched.append((rule, *found))

    level = next((lv for lv in LEVELS if any(m[0]["level"] == lv for m in matched)), None)
    top = [m for m in matched if m[0]["level"] == level]
    reasons = [
        localize(rule["reason"], lang).format(
            symptoms=", ".join(label(s) for s in symptoms),
            sign=", ".join(describe(image) for image in images),
        )
        for rule, symptoms, images, _ in top
    ]
    used_hint = next((images[0] for _, _, images, via in top if via), None)
    if used_hint:
        reasons.append(t("reason_uncertain_hint", lang).format(sign=describe(used_hint)))
    # The most urgent destination among the rules at the top level.
    destination = min(
        (m[0]["destination"] for m in top), key=DESTINATIONS.index, default=None
    )
    rules = [m[0]["id"] for m in matched]
    if level is None:
        # Context items (e.g. contact lenses) alone are not symptoms.
        symptoms_only = selected - CONTEXT_SYMPTOMS
        if symptoms_only or signs or display_only_signs(prediction):
            # Green needs the whole checklist reviewed and a known age group.
            if confirmed and age_group != "unknown":
                level, destination = "green", "self_care"
                if symptoms_only:
                    reasons.append(
                        t("reason_mild", lang).format(
                            symptoms=", ".join(
                                label(s) for s in SYMPTOMS if s in symptoms_only
                            )
                        )
                    )
            else:
                rules = ["needs_confirmation"]
    shown_only = display_only_signs(prediction)
    if shown_only:
        reasons.append(
            t("reason_image_display_only", lang).format(sign=", ".join(describe(s) for s in shown_only))
        )
    elif level and not signs and not used_hint:
        reasons.append(t("reason_image_not_used", lang))
    first_aid = [
        t(f"first_aid_{key}", lang)
        for key in FIRST_AID
        if any(key in m[0].get("first_aid", []) for m in matched)
    ]
    candidates = []
    for disease, symptoms in DISEASES:
        matches = sorted(selected & symptoms)
        image_match = sorted(signs & set(disease["image_signs"]), key=list(KB["image_signs"]).index)
        if not matches and not image_match:
            continue
        candidates.append(
            {
                "name": localize(disease["name"], lang),
                "match_points": len(matches) + 2 * len(image_match),
                "matched_symptoms": [label(s) for s in matches],
                "matched_image_sign": ", ".join(image_match) or None,
                "note": localize(disease["note"], lang),
            }
        )
    candidates.sort(key=lambda d: (-d["match_points"], d["name"]))
    if level:
        advice = level
    elif rules == ["needs_confirmation"]:
        advice = "confirm"
    else:
        advice = "none"
    return {
        "level": level,
        "label": t(f"level_{level or 'none'}", lang),
        "urgent": level == "red",
        "destination": destination,
        "destination_text": t(f"dest_{destination}", lang) if destination else None,
        "reasons": reasons,
        "recommendation": t(f"advice_{advice}", lang),
        "first_aid": first_aid,
        "warnings": [t("warning_steroids", lang)] if level else [],
        "rules": rules,
        "candidates": candidates[:3],
        "image_used": bool(signs),
        "knowledge_version": KB["version"],
        "disclaimer": localize(KB["disclaimer"], lang),
    }
