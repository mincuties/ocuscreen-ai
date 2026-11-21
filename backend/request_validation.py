"""Bound upload memory and validate multipart fields before inference."""

import asyncio
import json

from fastapi import HTTPException, Request

from backend.i18n import DEFAULT, t
from backend.image_predictor import MAX_BYTES
from backend.risk_engine import AGE_GROUPS, DURATIONS, SYMPTOMS

MAX_REQUEST_BYTES = MAX_BYTES + 65536
UPLOAD_TIMEOUT_SECONDS = 30


async def buffer_upload(request: Request, lang: str = DEFAULT) -> None:
    """Bound chunked requests too; Content-Length alone is not trustworthy."""
    media_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if media_type != "multipart/form-data":
        raise HTTPException(415, t("not_multipart", lang))
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise HTTPException(400, t("bad_content_length", lang))
        digits = content_length.lstrip("0") or "0"
        if len(digits) > len(str(MAX_REQUEST_BYTES)) or int(digits) > MAX_REQUEST_BYTES:
            raise HTTPException(413, t("too_large", lang))
    chunks, size = [], 0
    try:
        async with asyncio.timeout(UPLOAD_TIMEOUT_SECONDS):
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise HTTPException(413, t("too_large", lang))
                chunks.append(chunk)
    except TimeoutError as exc:
        raise HTTPException(408, t("upload_timeout", lang)) from exc
    # Starlette's form parser consumes this bounded body through request.stream().
    request._body = b"".join(chunks)


def parse_symptoms(raw: str, lang: str = DEFAULT) -> list[str]:
    try:
        ids = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, t("symptoms_not_json", lang)) from exc
    if (
        not isinstance(ids, list)
        or len(ids) > len(SYMPTOMS)
        or any(not isinstance(item, str) or item not in SYMPTOMS for item in ids)
    ):
        raise HTTPException(422, t("symptoms_invalid", lang))
    return list(dict.fromkeys(ids))


def parse_flag(form, key: str, lang: str = DEFAULT) -> bool:
    """An optional "true"/"false" field, false when missing or empty."""
    value = form.get(key) or "false"
    if len(form.getlist(key)) > 1 or value not in ("true", "false"):
        raise HTTPException(422, t("triage_field_invalid", lang))
    return value == "true"


def parse_triage(form, lang: str = DEFAULT) -> dict:
    """Optional answers that refine triage; a missing field means "not given"."""
    values = {key: form.get(key) or None for key in ("age_group", "duration")}
    if (
        any(len(form.getlist(key)) > 1 for key in ("age_group", "duration"))
        or values["age_group"] not in (None, *AGE_GROUPS)
        or values["duration"] not in (None, *DURATIONS)
    ):
        raise HTTPException(422, t("triage_field_invalid", lang))
    return {**values, "confirmed": parse_flag(form, "confirmed", lang)}
