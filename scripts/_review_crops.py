"""Review view for the bbox crops produced by :mod:`group_pipeline`.

Each group renders as a row showing every crop alongside its source
photo. The reviewer can mark each crop as reviewed (`✅`), flag it for
information excluded by the bbox (`✂`), and reference other crops in
the same group where the missing info IS visible (`↪`). Per-group
comments live next to the row (`💬`).

The crops to display come from each group's manifest written by
:mod:`group_pipeline`; persistence for the human review state lives
at :data:`DEFAULT_CROP_REVIEWS_JSON`. The current implementation
filters to roles in :data:`REVIEWED_ROLES` (front / nutrition /
price-tag) — other roles (ingredients, other-label) ship later.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
)

from _json_types import JsonObject, JsonValue
from _review_views import load_json_object, register_image_routes

REPO_ROOT = Path(__file__).resolve().parents[1]
URL_PREFIX = "crops"

DEFAULT_CROP_REVIEWS_JSON = REPO_ROOT / "data" / "crop_reviews.json"

# Roles that participate in the current review pass. Ingredients /
# other-label are produced by the pipeline but the user isn't checking
# them yet; UPC barcode panels are not produced at all. Adding a role
# here is enough to start reviewing it.
REVIEWED_ROLES = ("front", "nutrition", "price-tag")

# Display order within a group: roles cluster top-to-bottom in this
# order, then crops within a role go by capture time (filename sort).
ROLE_DISPLAY_ORDER = ("front", "nutrition", "price-tag")

ID_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"

INDEX_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Crop review</title>
<style>
body { font-family: system-ui, sans-serif; margin: 1em; background: #fafafa; }
.legend { background: #fffbea; border: 1px solid #ecc94b; border-radius: 6px;
          padding: 0.6em 1em; margin: 0.6em 0; font-size: 0.9em; }
.legend dt { font-weight: bold; display: inline-block; min-width: 1.5em; }
.legend dd { display: inline; margin: 0 1em 0 0.2em; }
.instructions { background: #ebf8ff; border: 1px solid #63b3ed; border-radius: 6px;
                padding: 0.6em 1em; margin: 0.6em 0; font-size: 0.9em; }
.group { background: white; border: 1px solid #ddd; border-radius: 6px;
         padding: 1em; margin-bottom: 1em; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.group-header { display: flex; justify-content: space-between; align-items: baseline; }
.group-id { font-weight: bold; }
.warnings { color: #c53030; font-size: 0.9em; }
.role-section { margin: 0.5em 0; }
.role-label { font-weight: 500; color: #1a365d; margin: 0.4em 0 0.2em 0; }
.crop { display: flex; align-items: flex-start; gap: 0.6em; margin: 0.4em 0;
        padding: 0.4em; border: 1px solid #eee; border-radius: 4px; }
.crop.reviewed { background: #f0fff4; border-color: #9ae6b4; }
.crop.excluded { background: #fff5f5; border-color: #fc8181; }
.crop-id { font-family: ui-monospace, monospace; font-weight: bold;
           font-size: 1.1em; min-width: 1.5em; color: #2c5282; }
.crop-images { display: flex; gap: 0.4em; }
.crop-image { text-align: center; font-size: 0.7em; color: #555; }
.crop-image a { display: block; }
.crop-image img { display: block; max-height: 160px; border-radius: 3px; }
.crop-image .label { font-weight: 500; }
.crop-image .filename { font-family: ui-monospace, monospace; font-size: 0.6em;
                        color: #444; word-break: break-all; max-width: 220px; }
.crop-image .image-missing { display: flex; flex-direction: column;
                             align-items: center; justify-content: center;
                             width: 220px; height: 160px;
                             background: #fff5f5; border: 2px dashed #c53030;
                             border-radius: 3px; padding: 0.4em;
                             color: #9b2c2c; font-size: 0.75em; }
.crop-image .image-missing strong { font-size: 0.95em; margin-bottom: 0.3em; }
.crop-controls { flex: 1; display: flex; flex-direction: column; gap: 0.3em; }
.crop-controls label { font-size: 0.9em; }
.crop-controls textarea { width: 100%; min-height: 1.8em; }
.crop-controls textarea.hidden { display: none; }
.group-comment { width: 100%; min-height: 2.4em; margin-top: 0.4em; }
.saved { color: #2f855a; font-size: 0.85em; opacity: 0; transition: opacity 0.3s; }
.saved.show { opacity: 1; }
</style>
</head><body>
<h1>Crop review — {{ groups|length }} groups</h1>

{% include 'instructions' ignore missing %}
<div class="instructions">
  <strong>How to use this page:</strong>
  Walk each group top-to-bottom. For every crop, check ✅ once you've looked at
  it. If the bbox cut off any text the reader will need (cropped past a curved
  bottle edge, missed the right side of a multi-panel page, hand-instead-of-
  panel), check ✂ — then if the missing text is visible in another crop in
  the SAME group, list those crops' ids in ↪ (comma-separated). Leave ↪ blank
  to assert the info is gone for good. Use 💬 for anything else worth noting.
  Saves on change; no submit button.
</div>

<div class="legend">
  <dt>✅</dt><dd>this crop has been reviewed</dd>
  <dt>✂</dt><dd>part of the desired information was cut off in this crop</dd>
  <dt>↪</dt><dd>ids of crops where the missing info IS visible (blank = lost)</dd>
  <dt>💬</dt><dd>group-level comment for anything the checkboxes don't capture</dd>
</div>

{% for g in groups %}
<div class="group" data-id="{{ g.id }}">
  <div class="group-header">
    <div>
      <span class="group-id">{{ g.id }}</span>
      <span>· {{ g.store }} · {{ g.crop_count }} crop{{ '' if g.crop_count == 1 else 's' }}</span>
    </div>
    <div class="warnings">{{ g.warnings|join('; ') }}</div>
  </div>

  {% for role in g.roles_in_order %}
    {% if g.crops_by_role[role] %}
    <div class="role-section">
      <div class="role-label">{{ role }}</div>
      {% for c in g.crops_by_role[role] %}
        <div class="crop {% if c.reviewed %}reviewed{% endif %} {% if c.info_excluded %}excluded{% endif %}"
             data-crop-key="{{ c.key }}">
          <div class="crop-id">{{ c.id }}</div>
          <div class="crop-images">
            <div class="crop-image">
              <div class="label">crop</div>
              {% if c.crop_missing %}
                <div class="image-missing" title="{{ c.crop_abs_path }}">
                  <strong>⚠ crop missing</strong>
                  <span>manifest references a file that isn't on disk; re-run group_pipeline.py to refresh</span>
                </div>
              {% else %}
                <a href="/crops/raw/crop/{{ g.id }}/{{ c.crop_filename }}" target="_blank" rel="noopener">
                  <img src="/crops/thumb/crop/{{ g.id }}/{{ c.crop_filename }}" alt="{{ c.crop_filename }}">
                </a>
              {% endif %}
              <div class="filename" title="{{ c.crop_abs_path }}">{{ c.crop_filename }}</div>
            </div>
            <div class="crop-image">
              <div class="label">source</div>
              {% if c.source_missing %}
                <div class="image-missing" title="{{ c.source_abs_path }}">
                  <strong>⚠ source missing</strong>
                  <span>manifest references a file that isn't on disk; re-run group_pipeline.py to refresh</span>
                </div>
              {% else %}
                <a href="/crops/raw/source/{{ g.id }}/{{ c.source_filename }}" target="_blank" rel="noopener">
                  <img src="/crops/thumb/source/{{ g.id }}/{{ c.source_filename }}" alt="{{ c.source_filename }}">
                </a>
              {% endif %}
              <div class="filename" title="{{ c.source_abs_path }}">{{ c.source_filename }}</div>
            </div>
          </div>
          <div class="crop-controls">
            <label>
              ✅ <input type="checkbox" class="reviewed-flag" autocomplete="off"
                data-server-checked="{{ '1' if c.reviewed else '0' }}"
                {% if c.reviewed %}checked{% endif %}>
              reviewed
            </label>
            <label>
              ✂ <input type="checkbox" class="excluded-flag" autocomplete="off"
                data-server-checked="{{ '1' if c.info_excluded else '0' }}"
                {% if c.info_excluded %}checked{% endif %}>
              info excluded
            </label>
            <textarea class="info-available-in {% if not c.info_excluded %}hidden{% endif %}"
              placeholder="↪ comma-separated crop ids where the missing info is visible"
              autocomplete="off"
              data-server-value="{{ c.info_available_in }}">{{ c.info_available_in }}</textarea>
            <span class="saved">saved</span>
          </div>
        </div>
      {% endfor %}
    </div>
    {% endif %}
  {% endfor %}

  <textarea class="group-comment" placeholder="💬 group comment (optional)"
    autocomplete="off"
    data-server-value="{{ g.comment }}">{{ g.comment }}</textarea>
  <span class="saved group-saved">saved</span>
</div>
{% endfor %}

<div class="legend">
  <dt>✅</dt><dd>this crop has been reviewed</dd>
  <dt>✂</dt><dd>part of the desired information was cut off in this crop</dd>
  <dt>↪</dt><dd>ids of crops where the missing info IS visible (blank = lost)</dd>
  <dt>💬</dt><dd>group-level comment for anything the checkboxes don't capture</dd>
</div>

<div class="instructions">
  <strong>Reminder:</strong>
  ✅ marks reviewed crops; ✂ flags crops where info was cut off; ↪ lists crop
  ids in this group where the missing info is still visible (blank = lost
  for good); 💬 captures group-level comments. Saves on change.
</div>

<script>
function syncFromServerState() {
  document.querySelectorAll('.crop').forEach(div => {
    const reviewed = div.querySelector('.reviewed-flag');
    const excluded = div.querySelector('.excluded-flag');
    const text = div.querySelector('.info-available-in');
    reviewed.checked = reviewed.dataset.serverChecked === '1';
    excluded.checked = excluded.dataset.serverChecked === '1';
    text.value = text.dataset.serverValue;
    div.classList.toggle('reviewed', reviewed.checked);
    div.classList.toggle('excluded', excluded.checked);
    text.classList.toggle('hidden', !excluded.checked);
  });
  document.querySelectorAll('.group-comment').forEach(t => {
    t.value = t.dataset.serverValue;
  });
}
window.addEventListener('pageshow', syncFromServerState);
syncFromServerState();

document.querySelectorAll('.crop').forEach(div => {
  const groupDiv = div.closest('.group');
  const groupId = groupDiv.dataset.id;
  const cropKey = div.dataset.cropKey;
  const reviewed = div.querySelector('.reviewed-flag');
  const excluded = div.querySelector('.excluded-flag');
  const text = div.querySelector('.info-available-in');
  const saved = div.querySelector('.saved');
  const send = () => {
    fetch('/crops/api/crop/' + groupId + '/' + encodeURIComponent(cropKey), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        reviewed: reviewed.checked,
        info_excluded: excluded.checked,
        info_available_in: text.value,
      }),
    }).then(r => {
      if (r.ok) {
        reviewed.dataset.serverChecked = reviewed.checked ? '1' : '0';
        excluded.dataset.serverChecked = excluded.checked ? '1' : '0';
        text.dataset.serverValue = text.value;
        div.classList.toggle('reviewed', reviewed.checked);
        div.classList.toggle('excluded', excluded.checked);
        text.classList.toggle('hidden', !excluded.checked);
        saved.classList.add('show');
        setTimeout(() => saved.classList.remove('show'), 600);
      }
    });
  };
  reviewed.addEventListener('change', send);
  excluded.addEventListener('change', send);
  text.addEventListener('change', send);
});

document.querySelectorAll('.group-comment').forEach(t => {
  const groupDiv = t.closest('.group');
  const groupId = groupDiv.dataset.id;
  const saved = groupDiv.querySelector('.group-saved');
  t.addEventListener('change', () => {
    fetch('/crops/api/group/' + groupId, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({comment: t.value}),
    }).then(r => {
      if (r.ok) {
        t.dataset.serverValue = t.value;
        saved.classList.add('show');
        setTimeout(() => saved.classList.remove('show'), 600);
      }
    });
  });
});
</script>
</body></html>
"""


def register(
    app: Flask,
    *,
    gold_groups_json: Path,
    stitched_root: Path,
    crop_reviews_json: Path = DEFAULT_CROP_REVIEWS_JSON,
) -> None:
    """Register the crops review view on ``app``."""

    def resolve_image(key: str) -> Path | None:
        # key has the form "crop/<group_id>/<filename>" or
        # "source/<group_id>/<filename>"; the kind tells us which
        # bucket to look in within the manifest.
        parts = key.split("/", 2)
        if len(parts) != 3:
            return None
        kind, group_id, filename = parts
        manifest = _load_manifest(stitched_root, group_id)
        if manifest is None:
            return None
        entries = manifest.get("crops")
        if not isinstance(entries, list):
            return None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            crop_path = entry.get("crop")
            source_path = entry.get("source_photo")
            if not isinstance(crop_path, str) or not isinstance(
                source_path, str
            ):
                continue
            target = (
                crop_path
                if kind == "crop"
                else source_path
                if kind == "source"
                else None
            )
            if target is None:
                return None
            if Path(target).name != filename:
                continue
            candidate = Path(target)
            resolved = (
                candidate
                if candidate.is_absolute()
                else (REPO_ROOT / candidate)
            )
            # Intentionally NO fallback: a manifest path that doesn't
            # resolve is a data problem (typically a stale manifest)
            # and the user needs to see it as an error rather than
            # have us paper over it with a guess. The page render
            # surfaces this explicitly via ``source_missing`` so the
            # bad rows are visible at a glance.
            return resolved if resolved.exists() else None
        return None

    register_image_routes(
        app,
        url_prefix=URL_PREFIX,
        resolver=resolve_image,
        endpoint_prefix="crops",
    )

    @app.get(f"/{URL_PREFIX}/", endpoint="crops_index")
    def index() -> str:
        groups = _build_render_groups(
            gold_groups_json=gold_groups_json,
            stitched_root=stitched_root,
            crop_reviews_json=crop_reviews_json,
        )
        return render_template_string(INDEX_TEMPLATE, groups=groups)

    @app.post(
        f"/{URL_PREFIX}/api/crop/<group_id>/<path:crop_key>",
        endpoint="crops_api_update_crop",
    )
    def update_crop(group_id: str, crop_key: str) -> Response:
        payload = request.get_json(silent=True) or {}
        reviewed = bool(payload.get("reviewed", False))
        info_excluded = bool(payload.get("info_excluded", False))
        info_available_in = str(payload.get("info_available_in", ""))
        _update_crop_state(
            crop_reviews_json,
            group_id,
            crop_key,
            reviewed=reviewed,
            info_excluded=info_excluded,
            info_available_in=info_available_in,
        )
        return jsonify({"ok": True})

    @app.post(
        f"/{URL_PREFIX}/api/group/<group_id>",
        endpoint="crops_api_update_group",
    )
    def update_group(group_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        comment = str(payload.get("comment", ""))
        _update_group_comment(crop_reviews_json, group_id, comment)
        return jsonify({"ok": True})


def encode_crop_id(index: int) -> str:
    """Encode a per-group crop index as a base-36 short id.

    Single character for indices 0–35, then ``10``, ``11``, ..., ``zz``,
    ``100``, etc. — natural extension of the user's "0-9, a-z, multiple
    chars beyond 36" rule.
    """
    if index < 0:
        raise ValueError(f"crop index must be >= 0, got {index}")
    if index == 0:
        return "0"
    out: list[str] = []
    n = index
    while n:
        n, r = divmod(n, 36)
        out.append(ID_DIGITS[r])
    return "".join(reversed(out))


def _build_render_groups(
    *,
    gold_groups_json: Path,
    stitched_root: Path,
    crop_reviews_json: Path,
) -> list[JsonObject]:
    """Build the per-group dicts the template iterates over.

    Combines gold groups (for ordering and store) with each group's
    manifest (for the canonical crop list) and the persisted review
    state (for restoring checkbox/textbox values).
    """
    gold = load_json_object(gold_groups_json) or {}
    raw_groups = gold.get("groups") if isinstance(gold, dict) else None
    if not isinstance(raw_groups, list):
        return []

    reviews = _load_crop_reviews(crop_reviews_json)
    out: list[JsonObject] = []
    for g in raw_groups:
        if not isinstance(g, dict):
            continue
        gid = g.get("id")
        if not isinstance(gid, str):
            continue
        store_value = g.get("store")
        store = store_value if isinstance(store_value, str) else ""

        per_group_review: JsonObject | None = reviews.get(gid)
        if not isinstance(per_group_review, dict):
            per_group_review = None
        crops_review_state = (
            per_group_review.get("crops")
            if per_group_review is not None
            else None
        )
        comment_value = (
            per_group_review.get("comment")
            if per_group_review is not None
            else ""
        )
        comment = comment_value if isinstance(comment_value, str) else ""

        crops_by_role: dict[str, list[JsonValue]] = defaultdict(list)
        crop_count = 0
        manifest = _load_manifest(stitched_root, gid)
        if manifest is not None:
            for entry in _ordered_manifest_entries(manifest):
                role = entry.get("role")
                crop_path = entry.get("crop")
                source_path = entry.get("source_photo")
                if (
                    not isinstance(role, str)
                    or role not in REVIEWED_ROLES
                    or not isinstance(crop_path, str)
                    or not isinstance(source_path, str)
                ):
                    continue
                crops_by_role[role].append(
                    _crop_render_entry(
                        crop_path=crop_path,
                        source_path=source_path,
                        review_state=crops_review_state,
                    )
                )
                crop_count += 1

        # Assign per-group base-36 short ids in role-major, capture-
        # time-minor order so the printed labels match the rendered
        # ordering. Non-reviewed roles aren't displayed; their entries
        # don't consume id slots.
        index = 0
        for role in ROLE_DISPLAY_ORDER:
            for entry in crops_by_role[role]:
                if isinstance(entry, dict):
                    entry["id"] = encode_crop_id(index)
                    index += 1

        warnings: list[JsonValue] = [
            f"missing {role}"
            for role in ROLE_DISPLAY_ORDER
            if not crops_by_role[role]
        ]
        roles_in_order: list[JsonValue] = list(ROLE_DISPLAY_ORDER)

        # Build the rendered dict in pieces so pyrefly's invariant-list
        # checks don't reject e.g. list[str] where list[JsonValue] is
        # expected. JsonObject's `dict[str, JsonValue]` value-type
        # position would otherwise force every contributing list to be
        # widened up front.
        rendered: JsonObject = {
            "id": gid,
            "store": store,
            "warnings": warnings,
            "crop_count": crop_count,
            "roles_in_order": roles_in_order,
            "comment": comment,
            "crops_by_role": {
                role: list(crops) for role, crops in crops_by_role.items()
            },
        }
        out.append(rendered)
    return out


def _ordered_manifest_entries(manifest: JsonObject) -> list[JsonObject]:
    """Pull out the crop entries in display order: role then capture time.

    Sorts by ``role``, ``source_photo``, then ``crop``. Source-photo
    filename encodes the timestamp, so this matches capture time within
    each role. Non-dict entries are dropped at this layer.
    """
    entries = manifest.get("crops")
    if not isinstance(entries, list):
        return []
    return sorted(
        (e for e in entries if isinstance(e, dict)),
        key=lambda e: (
            str(e.get("role", "")),
            str(e.get("source_photo", "")),
            str(e.get("crop", "")),
        ),
    )


def _crop_render_entry(
    *,
    crop_path: str,
    source_path: str,
    review_state: object,
) -> JsonObject:
    """Build the dict the template needs for one crop row.

    Includes ``crop_missing`` / ``source_missing`` flags computed from
    the manifest's recorded paths. The template renders a visible
    error in place of the image when either is true so a stale
    manifest can't hide as a broken-image icon — the user needs to
    see the data problem and fix it (typically by re-running
    ``group_pipeline.py`` to refresh the manifest).
    """
    crop_filename = Path(crop_path).name
    source_filename = Path(source_path).name
    per_crop: JsonObject = {}
    if isinstance(review_state, dict):
        candidate = review_state.get(crop_filename)
        if isinstance(candidate, dict):
            per_crop = candidate
    crop_resolved = REPO_ROOT / crop_path
    source_resolved = REPO_ROOT / source_path
    return {
        "key": crop_filename,
        "crop_filename": crop_filename,
        "source_filename": source_filename,
        "crop_abs_path": str(crop_resolved.resolve()),
        "source_abs_path": str(source_resolved.resolve()),
        "crop_missing": not crop_resolved.exists(),
        "source_missing": not source_resolved.exists(),
        "reviewed": bool(per_crop.get("reviewed", False)),
        "info_excluded": bool(per_crop.get("info_excluded", False)),
        "info_available_in": str(per_crop.get("info_available_in", "")),
    }


def _load_manifest(stitched_root: Path, group_id: str) -> JsonObject | None:
    """Read the per-group manifest written by group_pipeline."""
    return load_json_object(stitched_root / group_id / "manifest.json")


def _load_crop_reviews(crop_reviews_json: Path) -> dict[str, JsonObject]:
    """Load the persisted crop review state, keyed by group id."""
    payload = load_json_object(crop_reviews_json)
    if payload is None:
        return {}
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        return {}
    out: dict[str, JsonObject] = {}
    for gid, state in groups.items():
        if isinstance(gid, str) and isinstance(state, dict):
            out[gid] = state
    return out


def crops_flagged_info_lost(
    crop_reviews_json: Path, group_id: str
) -> frozenset[str]:
    """Return crop filenames in ``group_id`` flagged "info lost".

    "Info lost" means the reviewer marked the crop ✂ (info_excluded=True)
    AND left the recovery pointer (info_available_in) blank — i.e. no
    other crop in the same group has the missing text. Downstream
    extraction can use this to augment the image set with the
    uncropped source photo of each flagged crop, recovering text the
    crop dropped.

    Returns an empty set if ``crop_reviews_json`` is missing or the
    group has no review state — extraction continues with crops only.
    """
    reviews = _load_crop_reviews(crop_reviews_json)
    state = reviews.get(group_id)
    if not isinstance(state, dict):
        return frozenset()
    crops = state.get("crops")
    if not isinstance(crops, dict):
        return frozenset()
    out: set[str] = set()
    for filename, per_crop in crops.items():
        if not isinstance(filename, str) or not isinstance(per_crop, dict):
            continue
        if per_crop.get("info_excluded") and not per_crop.get(
            "info_available_in"
        ):
            out.add(filename)
    return frozenset(out)


def _update_crop_state(
    crop_reviews_json: Path,
    group_id: str,
    crop_key: str,
    *,
    reviewed: bool,
    info_excluded: bool,
    info_available_in: str,
) -> None:
    """Save the per-crop review state, creating the file if missing."""
    payload = _load_or_init_crop_reviews(crop_reviews_json)
    groups = _ensure_dict(payload, "groups")
    group_state = _ensure_dict(groups, group_id)
    crops = _ensure_dict(group_state, "crops")
    crops[crop_key] = {
        "reviewed": reviewed,
        "info_excluded": info_excluded,
        "info_available_in": info_available_in,
    }
    crop_reviews_json.parent.mkdir(parents=True, exist_ok=True)
    crop_reviews_json.write_text(json.dumps(payload, indent=2))


def _update_group_comment(
    crop_reviews_json: Path, group_id: str, comment: str
) -> None:
    """Save the per-group comment, creating the file if missing."""
    payload = _load_or_init_crop_reviews(crop_reviews_json)
    groups = _ensure_dict(payload, "groups")
    group_state = _ensure_dict(groups, group_id)
    group_state["comment"] = comment
    crop_reviews_json.parent.mkdir(parents=True, exist_ok=True)
    crop_reviews_json.write_text(json.dumps(payload, indent=2))


def _load_or_init_crop_reviews(crop_reviews_json: Path) -> JsonObject:
    """Load the file if present, else return a fresh shell."""
    if not crop_reviews_json.exists():
        return {"groups": {}}
    payload: JsonValue = json.loads(crop_reviews_json.read_text())
    if not isinstance(payload, dict):
        return {"groups": {}}
    return payload


def _ensure_dict(parent: JsonObject, key: str) -> JsonObject:
    """Get-or-create a nested dict at ``parent[key]``."""
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value
