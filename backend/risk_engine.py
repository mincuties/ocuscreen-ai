"""Transparent prototype rules. No trained disease probabilities."""

import json
from pathlib import Path

from backend.i18n import DEFAULT, localize, t

KB = json.loads(
    (Path(__file__).resolve().parents[1] / "data/disease_knowledge.json").read_text(
        encoding="utf-8"
    )
)
SYMPTOMS = {s["id"]: s for s in KB["symptoms"]}
URGENT_SYMPTOMS = frozenset(KB["urgent_symptoms"])
COMBINATIONS = [
    (rule, frozenset(rule["all_symptoms"])) for rule in KB["urgent_combinations"]
]
DISEASES = [(disease, frozenset(disease["symptoms"])) for disease in KB["diseases"]]


def localized_symptoms(lang=DEFAULT):
    return [{**s, "label": localize(s["label"], lang)} for s in KB["symptoms"]]


def localized_combinations(lang=DEFAULT):
    return [
        {**rule, "reason": localize(rule["reason"], lang)}
        for rule in KB["urgent_combinations"]
    ]


def assess_risk(prediction, symptom_ids, lang=DEFAULT):
    selected = set(symptom_ids)
    if selected - SYMPTOMS.keys():
        raise ValueError("Unknown symptom ID.")

    def label(sid):
        return localize(SYMPTOMS[sid]["label"], lang)

    sign = prediction["label"] if prediction["status"] == "valid" else None
    reasons = []
    high = selected & URGENT_SYMPTOMS
    for sid in sorted(high):
        reasons.append(label(sid))
    for rule, required in COMBINATIONS:
        if required <= selected and (
            not rule["image_signs"] or sign in rule["image_signs"]
        ):
            high.add(rule["id"])
            reasons.append(localize(rule["reason"], lang))
    image_rule = KB["image_signs"].get(sign, {})
    level = (
        "high"
        if high
        else ("medium" if selected or (sign and sign != "Normal") else "low")
    )
    if not high and prediction["status"] != "valid" and not selected:
        level = None
    if image_rule.get("priority") == "high":
        level = "high"
        reasons.append(localize(image_rule["description"], lang))
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
    advice = "urgent" if high else (level or "none")
    return {
        "level": level,
        "label": t(f"level_{level or 'none'}", lang),
        "urgent": bool(high),
        "reasons": reasons,
        "recommendation": t(f"advice_{advice}", lang),
        "candidates": candidates[:3],
        "image_used": sign is not None,
        "knowledge_version": KB["version"],
        "disclaimer": localize(KB["disclaimer"], lang),
    }
