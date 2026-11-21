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
CONTEXT_SYMPTOMS = frozenset(s["id"] for s in KB["symptoms"] if s.get("context"))
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
    # Context items (e.g. wearing contact lenses) count in combinations, not alone.
    symptoms_only = selected - CONTEXT_SYMPTOMS
    level = (
        "high"
        if high
        else ("medium" if symptoms_only or (sign and sign != "Normal") else "low")
    )
    if not high and prediction["status"] != "valid" and not symptoms_only:
        level = None
    # An uncertain image that leans towards a sign is not dropped silently when the
    # sign is high priority, or would complete a combination rule with the selected
    # symptoms (e.g. contact lenses + redness). It nudges towards an exam: never to
    # urgent, never into the rules or candidates.
    hint = prediction.get("hint") or {}
    hint_label = hint.get("label")
    hint_rule = KB["image_signs"].get(hint_label, {})
    completes_rule = any(
        required <= selected and hint_label in rule["image_signs"]
        for rule, required in COMBINATIONS
    )
    if not high and hint_rule and (hint_rule.get("priority") == "high" or completes_rule):
        level = "medium" if level in (None, "low") else level
        reasons.append(
            t("reason_uncertain_hint", lang).format(
                sign=localize(hint_rule["description"], lang)
            )
        )
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
