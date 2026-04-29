"""Review view for ``data/groups.json`` — the photo-grouping output.

Each group renders as a row with thumbnails of its raw photos plus an
"errors" checkbox and a comment box. Submitting POSTs back to
``/groups/api/<id>`` which updates the on-disk JSON in place.

This is the original review use case (Phase A grouping). The crop
review view in :mod:`_review_crops` is its sibling — both share image
serving via :mod:`_review_views`.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
)

from _json_types import JsonObject, JsonValue
from _review_views import load_json_object, register_image_routes
from group_photos import _filename_time_delta_seconds

REPO_ROOT = Path(__file__).resolve().parents[1]
URL_PREFIX = "groups"

# A short gap between groups means the boundary is more likely to be
# wrong (the photographer might still be documenting the same product).
SHORT_GAP_SECONDS = 30.0

INDEX_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Photo group review</title>
<style>
body { font-family: system-ui, sans-serif; margin: 1em; background: #fafafa; }
.group { background: white; border: 1px solid #ddd; border-radius: 6px;
         padding: 1em; margin-bottom: 1em; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.group.has-errors { border-color: #c53030; background: #fff5f5; }
.gap { font-size: 0.8em; color: #666; margin: 0.6em 0 0.3em 0.5em; }
.gap.short { color: #c05621; font-weight: 500; }
.group-header { display: flex; justify-content: space-between; align-items: baseline; }
.group-id { font-weight: bold; }
.warnings { color: #c53030; font-size: 0.9em; }
.photos { display: flex; flex-wrap: wrap; gap: 0.6em; margin: 0.5em 0; }
.photo { text-align: center; font-size: 0.7em; color: #555; max-width: 200px; }
.photo a { display: block; }
.photo img { display: block; max-height: 180px; border-radius: 3px; }
.photo .filename { font-family: ui-monospace, monospace; font-size: 0.65em;
                   color: #444; word-break: break-all; margin-top: 0.2em; }
.photo .roles { font-weight: 500; color: #1a365d; }
.controls { display: flex; gap: 1em; align-items: center; margin-top: 0.5em; }
.controls textarea { flex: 1; min-height: 2.4em; }
.saved { color: #2f855a; font-size: 0.85em; margin-left: 0.5em; opacity: 0; transition: opacity 0.3s; }
.saved.show { opacity: 1; }
</style>
</head><body>
<h1>Group review — {{ groups|length }} groups</h1>
<p>Tick "errors" if a group is wrong. Comments help future-you. Saves on
change; no submit button.</p>

{% for g in groups %}
{% if g.gap_to_previous %}
<div class="gap {% if g.gap_short %}short{% endif %}">⏱ {{ g.gap_to_previous }} since previous group</div>
{% endif %}
<div class="group {% if g.has_errors %}has-errors{% endif %}" data-id="{{ g.id }}">
  <div class="group-header">
    <div>
      <span class="group-id">{{ g.id }}</span>
      <span>· {{ g.store }} · {{ g.photos|length }} photos</span>
    </div>
    <div class="warnings">{{ g.warnings|join('; ') }}</div>
  </div>
  <div class="photos">
    {% for p in g.photos %}
      <div class="photo">
        <a href="/groups/raw/{{ p.filename }}" target="_blank" rel="noopener">
          <img src="/groups/thumb/{{ p.filename }}" alt="{{ p.filename }}">
        </a>
        <div class="filename">{{ p.filename }}</div>
        <div class="roles">{{ p.roles|join(', ') }}</div>
      </div>
    {% endfor %}
  </div>
  <div class="controls">
    <label>
      <input type="checkbox" class="errors-flag" autocomplete="off"
        data-server-checked="{{ '1' if g.has_errors else '0' }}"
        {% if g.has_errors %}checked{% endif %}>
      Errors
    </label>
    <textarea class="comment" placeholder="comment (optional)" autocomplete="off"
      data-server-value="{{ g.comment }}">{{ g.comment }}</textarea>
    <span class="saved">saved</span>
  </div>
</div>
{% endfor %}

<script>
// Browsers' bfcache restores form input values across reloads/back-forward,
// overriding the server-rendered state we just sent. Re-sync from
// data-server-* attributes whenever the page becomes visible. Without this
// the checkbox/comment for the previously-reviewed session "follows" the
// user into a fresh review run.
function syncFromServerState() {
  document.querySelectorAll('.group').forEach(div => {
    const flag = div.querySelector('.errors-flag');
    const comment = div.querySelector('.comment');
    flag.checked = flag.dataset.serverChecked === '1';
    comment.value = comment.dataset.serverValue;
    div.classList.toggle('has-errors', flag.checked);
  });
}
window.addEventListener('pageshow', syncFromServerState);
syncFromServerState();

document.querySelectorAll('.group').forEach(div => {
  const id = div.dataset.id;
  const flag = div.querySelector('.errors-flag');
  const comment = div.querySelector('.comment');
  const saved = div.querySelector('.saved');
  const send = () => {
    fetch('/groups/api/' + id, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        has_errors: flag.checked,
        comment: comment.value,
      }),
    }).then(r => {
      if (r.ok) {
        div.classList.toggle('has-errors', flag.checked);
        flag.dataset.serverChecked = flag.checked ? '1' : '0';
        comment.dataset.serverValue = comment.value;
        saved.classList.add('show');
        setTimeout(() => saved.classList.remove('show'), 600);
      }
    });
  };
  flag.addEventListener('change', send);
  comment.addEventListener('change', send);
});
</script>
</body></html>
"""


def register(
    app: Flask,
    *,
    groups_json: Path,
    photos_dir: Path,
) -> None:
    """Register the groups review view on ``app``.

    ``photos_dir`` is the *fallback* directory for thumbnails — the
    actual lookup first tries to resolve the path stored in
    ``groups.json``. Photos arriving via the boundary-resolution
    extension pool carry repo-relative paths, and we want those served
    too.
    """

    def resolve_photo(filename: str) -> Path | None:
        groups_root = groups_json.resolve().parent
        for g in _load_groups(groups_json):
            for stored in _photo_paths_in_group(g):
                if stored.name != filename:
                    continue
                candidate = (
                    stored if stored.is_absolute() else (groups_root / stored)
                )
                if candidate.exists():
                    return candidate
                # Plan stores paths repo-relative to the project root
                # (data/raw_photos/...). Try that too.
                repo_relative = REPO_ROOT / stored
                if repo_relative.exists():
                    return repo_relative
        fallback = photos_dir / filename
        return fallback if fallback.exists() else None

    register_image_routes(
        app,
        url_prefix=URL_PREFIX,
        resolver=resolve_photo,
        endpoint_prefix="groups",
    )

    @app.get(f"/{URL_PREFIX}/", endpoint="groups_index")
    def index() -> str:
        groups = _annotate_groups_for_render(_load_groups(groups_json))
        return render_template_string(INDEX_TEMPLATE, groups=groups)

    @app.post(f"/{URL_PREFIX}/api/<group_id>", endpoint="groups_api_update")
    def update_group(group_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        has_errors = bool(payload.get("has_errors", False))
        comment = str(payload.get("comment", ""))
        if not _update_group_in_place(
            groups_json, group_id, has_errors, comment
        ):
            abort(404)
        return jsonify({"ok": True})


def format_gap(
    prev_filename: str | None, next_filename: str | None
) -> tuple[str | None, bool]:
    """Return (human-readable gap string, is_short_gap_flag) or (None, False).

    A short gap (under :data:`SHORT_GAP_SECONDS`) deserves visual
    attention since the previous group may have been split mid-product.
    """
    if prev_filename is None or next_filename is None:
        return None, False
    delta = _filename_time_delta_seconds(
        Path(prev_filename), Path(next_filename)
    )
    if delta is None:
        return None, False
    short = delta < SHORT_GAP_SECONDS
    if delta < 60:
        text = f"{delta:.0f} s"
    elif delta < 3600:
        m = int(delta // 60)
        s = int(delta - m * 60)
        text = f"{m} min {s} s"
    else:
        h = int(delta // 3600)
        m = int((delta - h * 3600) // 60)
        text = f"{h} h {m} min"
    return text, short


def _annotate_groups_for_render(groups: list[JsonObject]) -> list[JsonObject]:
    """Add the per-group fields the template needs.

    Mutates each group dict in place with ``filename`` on each photo
    plus ``has_errors``, ``comment``, ``gap_to_previous``, ``gap_short``
    on the group itself.
    """
    prev_last_filename: str | None = None
    for g in groups:
        photos = g.get("photos")
        if not isinstance(photos, list):
            photos = []
            g["photos"] = photos
        filenames: list[str] = []
        for p in photos:
            if not isinstance(p, dict):
                continue
            path_value = p.get("path")
            if not isinstance(path_value, str):
                continue
            name = Path(path_value).name
            p["filename"] = name
            filenames.append(name)
        g.setdefault("has_errors", False)
        g.setdefault("comment", "")
        first_filename = filenames[0] if filenames else None
        gap_str, gap_short = format_gap(prev_last_filename, first_filename)
        g["gap_to_previous"] = gap_str
        g["gap_short"] = gap_short
        if filenames:
            prev_last_filename = filenames[-1]
    return groups


def _load_groups(groups_json: Path) -> list[JsonObject]:
    """Load and shape-check the ``groups`` array from ``groups_json``."""
    payload = load_json_object(groups_json)
    if payload is None:
        return []
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return []
    return [g for g in groups if isinstance(g, dict)]


def _photo_paths_in_group(g: JsonObject) -> list[Path]:
    photos = g.get("photos")
    if not isinstance(photos, list):
        return []
    out: list[Path] = []
    for p in photos:
        if not isinstance(p, dict):
            continue
        path_value = p.get("path")
        if isinstance(path_value, str):
            out.append(Path(path_value))
    return out


def _update_group_in_place(
    groups_json: Path, group_id: str, has_errors: bool, comment: str
) -> bool:
    """Update one group's review-flag fields without touching anything else.

    Reads/writes the full JSON document so any extra fields (manual
    annotations, future schema additions) round-trip intact.
    """
    payload: JsonValue = json.loads(groups_json.read_text())
    if not isinstance(payload, dict):
        return False
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return False
    for g in groups:
        if not isinstance(g, dict):
            continue
        if g.get("id") == group_id:
            g["has_errors"] = has_errors
            g["comment"] = comment
            groups_json.write_text(json.dumps(payload, indent=2))
            return True
    return False
