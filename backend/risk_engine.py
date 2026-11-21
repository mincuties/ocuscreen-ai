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


def match(rule, selected, age_group, duration, sign, hint_label):
    """Return (matched symptoms, matched image sign, via hint) or None."""
    if not set(rule.get("all", [])) <= selected:
        return None
    if "any" in rule and not selected & set(rule["any"]):
        return None
    if "age" in rule and age_group not in rule["age"]:
        return None
    if "duration" in rule and duration not in rule["duration"]:
        return None
    image, via_hint = None, False
    if "image" in rule:
        if sign in rule["image"]:
            image = sign
        elif rule.get("hint_ok") and hint_label in rule["image"]:
            image, via_hint = hint_label, True
        else:
            return None
    # Name the symptoms that made an "any" rule fire; "all" ones are in its reason.
    named = selected & set(rule.get("any") or rule.get("all", []))
    return sorted(named, key=list(SYMPTOMS).index), image, via_hint


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

    sign = prediction["label"] if prediction["status"] == "valid" else None
    sign = sign if sign in KB["image_signs"] else None
    # An uncertain image can only count where a rule allows it (hint_ok).
    hint_label = (prediction.get("hint") or {}).get("label") if not sign else None
    matched = []
    for rule in RULES:
        found = match(rule, selected, age_group, duration, sign, hint_label)
        if found:
            matched.append((rule, *found))

    level = next((lv for lv in LEVELS if any(m[0]["level"] == lv for m in matched)), None)
    top = [m for m in matched if m[0]["level"] == level]
    reasons = [
        localize(rule["reason"], lang).format(
            symptoms=", ".join(label(s) for s in symptoms),
            sign=describe(image) if image else "",
        )
        for rule, symptoms, image, _ in top
    ]
    used_hint = next((image for _, _, image, via in top if via), None)
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
        if symptoms_only or sign:
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
    if level and not sign and not used_hint:
        reasons.append(t("reason_image_not_used", lang))
    first_aid = [
        t(f"first_aid_{key}", lang)
        for key in FIRST_AID
        if any(key in m[0].get("first_aid", []) for m in matched)
    ]
    candidates = []
    for disease, symptoms in DISEASES:
        matches = sorted(selected & symptoms)
        image_match = sign in disease["image_signs"]
        if not matches and not image_match:
            continue
        candidates.append(
            {
                "name": localize(disease["name"], lang),
                "match_points": len(matches) + 2 * int(image_match),
                "matched_symptoms": [label(s) for s in matches],
                "matched_image_sign": sign if image_match else None,
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
        "image_used": sign is not None,
        "knowledge_version": KB["version"],
        "disclaimer": localize(KB["disclaimer"], lang),
    }
