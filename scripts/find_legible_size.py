"""Per-photo binary-search for the smallest legible longest-side.

For each photo:

1. Use Opus 4.7 at full resolution to extract a reference payload (text +
   barcode digit strings). That's the ground truth.
2. Pick the cheapest model (Haiku 4.5 → Sonnet 4.6 → Opus 4.7) that
   matches the reference at full resolution. That becomes the "search
   model" — we use it for the binary-search probes because it's cheap.
3. Binary-search for the smallest longest-side at which the search model
   still extracts a payload that matches the reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from _claude import ClaudeRequest, ClaudeStructuredOutputError, call
from _image_ops import longest_side_px, resize_to_longest_side
from sizing_model import Probe

REPO_ROOT = Path(__file__).resolve().parents[1]
RESIZE_CACHE_DIR = REPO_ROOT / "data" / "cache" / "resized"

EXTRACTION_SYSTEM_PROMPT = (
    "You extract text from product photos for a deterministic comparison "
    "task. Given an image, return a JSON object with two keys: "
    '`"text"` (a single string of all visible text in the image, '
    "normalized to a single space between tokens, lower-cased, with "
    "punctuation preserved); and "
    '`"barcodes"` (a list of digit strings — every fully readable barcode '
    "you can see, with no spaces). If a barcode is partially obscured or "
    "ambiguous, omit it rather than guess. Return JSON only, no prose."
)

EXTRACTION_PROMPT = (
    "Extract the text and barcodes from this image, following the "
    "instructions in the system prompt."
)

# JSON schema for the extraction output. claude -p validates against this
# (when --json-schema is supported by the model) and we fall back to parsing
# fenced JSON blocks.
EXTRACTION_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "barcodes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text", "barcodes"],
        "additionalProperties": False,
    }
)


@dataclass(frozen=True)
class ExtractedPayload:
    """The model's structured view of a photo."""

    text: str  # normalized text
    barcodes: tuple[str, ...]  # digit strings


@dataclass
class SearchResult:
    """The outcome of a single-photo binary search."""

    photo_id: str
    reference_model: str
    search_model: str
    min_legible_size: int
    probes: list[Probe]


def extract_payload(
    image_path: Path, model: str, longest_side: int | None = None
) -> ExtractedPayload:
    """Run the extraction prompt on the (optionally resized) image."""
    if longest_side is None:
        target_path = image_path
    else:
        target_path = resize_to_longest_side(
            image_path, longest_side, RESIZE_CACHE_DIR
        )

    response = call(
        ClaudeRequest(
            prompt=EXTRACTION_PROMPT,
            model=model,
            image_paths=(target_path,),
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            json_schema=EXTRACTION_JSON_SCHEMA,
        )
    )
    return _parse_payload(response.text)


def matches_reference(
    reference: ExtractedPayload,
    candidate: ExtractedPayload,
    *,
    text_ratio_threshold: float = 0.95,
    require_barcodes: bool = False,
) -> bool:
    """Levenshtein-ratio match against the reference, with optional strict
    barcode parity.

    ``require_barcodes`` (default False): when True, every barcode digit-string
    in the reference must also be in the candidate — appropriate when the
    *barcode* itself is the artifact under test. For sizing decisions we
    care about text legibility, and barcode digits degrade ahead of the
    surrounding text under downsampling (a single dropped digit will fail
    a strict check while the rest of the label is still perfectly
    readable). The first live sizing run forced binary search up to
    full resolution on most photos because of this; with the default the
    binary search converges on a true text-legibility threshold.
    """
    text_ok = (
        _levenshtein_ratio(reference.text, candidate.text)
        >= text_ratio_threshold
    )
    if not text_ok:
        return False
    if require_barcodes:
        return all(b in candidate.barcodes for b in reference.barcodes)
    return True


def select_search_model(
    image_path: Path,
    reference: ExtractedPayload,
    candidates: tuple[str, ...] = ("haiku", "sonnet", "opus"),
) -> tuple[str, ExtractedPayload]:
    """Cheapest model whose full-res extraction matches ``reference``."""
    last_payload = reference
    for model in candidates:
        payload = extract_payload(image_path, model)
        if matches_reference(reference, payload):
            return model, payload
        last_payload = payload
    return candidates[-1], last_payload


def binary_search_min_size(
    image_path: Path,
    search_model: str,
    reference: ExtractedPayload,
    *,
    photo_id: str,
    lower_bound: int = 64,
) -> SearchResult:
    """Find the smallest longest-side at which ``search_model`` matches reference.

    Implements the user's "non-monotonic correction": after the search, any
    matching size with a larger failure-probe in the same run is treated as
    luck and skipped over.
    """
    upper = longest_side_px(image_path)
    if upper <= lower_bound:
        return SearchResult(
            photo_id=photo_id,
            reference_model="opus",
            search_model=search_model,
            min_legible_size=upper,
            probes=[Probe(photo_id=photo_id, size=upper, matched=True)],
        )

    lo, hi = lower_bound, upper
    probes: list[Probe] = []

    while lo < hi:
        mid = (lo + hi) // 2
        try:
            payload = extract_payload(
                image_path, search_model, longest_side=mid
            )
            matched = matches_reference(reference, payload)
        except ClaudeStructuredOutputError:
            # Model couldn't produce a valid extraction at this size — too
            # little text for it to satisfy the schema. That's effectively
            # "doesn't match the reference at this resolution," same
            # signal as if the extracted text were garbled.
            matched = False
        probes.append(Probe(photo_id=photo_id, size=mid, matched=matched))
        if matched:
            hi = mid
        else:
            lo = mid + 1

    failed_sizes = [p.size for p in probes if not p.matched]
    matched_sizes = sorted(p.size for p in probes if p.matched)
    final = upper
    for s in matched_sizes:
        if not any(f > s for f in failed_sizes):
            final = s
            break
    return SearchResult(
        photo_id=photo_id,
        reference_model="opus",
        search_model=search_model,
        min_legible_size=final,
        probes=probes,
    )


def _parse_payload(raw: str) -> ExtractedPayload:
    """Parse the model's response into an ExtractedPayload.

    Tolerates fenced ``` blocks and stray prose around the JSON.
    """
    cleaned = _extract_json_object(raw)
    data = json.loads(cleaned)
    text = str(data.get("text", "")).lower()
    barcodes = tuple(str(b) for b in data.get("barcodes", []))
    return ExtractedPayload(text=text, barcodes=barcodes)


def _extract_json_object(raw: str) -> str:
    """Pull the outermost JSON object out of ``raw``."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"No JSON object found in extraction response: {raw!r}"
        )
    return raw[start : end + 1]


def _levenshtein_ratio(a: str, b: str) -> float:
    """Return normalized similarity in [0, 1] (1 minus distance / longest)."""
    if not a and not b:
        return 1.0
    distance = _levenshtein(a, b)
    return 1.0 - distance / max(len(a), len(b))


def _levenshtein(a: str, b: str) -> int:
    """Compute DP edit distance — fine for the short normalized strings here."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[-1]
