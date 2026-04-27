"""Minimal Flask review UI for data/groups.json.

Renders one row per group with thumbnails of every photo in that group
plus a "this group has errors" checkbox and a comment box. Submitting the
form POSTs back to ``/api/group/<id>`` which updates ``data/groups.json``
in place.

Run:

    uv run python scripts/review_server.py [--port 8765]

Then open http://localhost:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    send_file,
)
from PIL import Image, ImageOps

from group_photos import GROUPS_JSON, RAW_PHOTOS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
THUMBNAIL_LONGEST_SIDE = 320

INDEX_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Photo group review</title>
<style>
body { font-family: system-ui, sans-serif; margin: 1em; background: #fafafa; }
.group { background: white; border: 1px solid #ddd; border-radius: 6px;
         padding: 1em; margin-bottom: 1em; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.group.has-errors { border-color: #c53030; background: #fff5f5; }
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
        <a href="/raw/{{ p.filename }}" target="_blank" rel="noopener">
          <img src="/thumb/{{ p.filename }}" alt="{{ p.filename }}">
        </a>
        <div class="filename">{{ p.filename }}</div>
        <div class="roles">{{ p.roles|join(', ') }}</div>
      </div>
    {% endfor %}
  </div>
  <div class="controls">
    <label>
      <input type="checkbox" class="errors-flag" {% if g.has_errors %}checked{% endif %}>
      Errors
    </label>
    <textarea class="comment" placeholder="comment (optional)">{{ g.comment }}</textarea>
    <span class="saved">saved</span>
  </div>
</div>
{% endfor %}

<script>
document.querySelectorAll('.group').forEach(div => {
  const id = div.dataset.id;
  const flag = div.querySelector('.errors-flag');
  const comment = div.querySelector('.comment');
  const saved = div.querySelector('.saved');
  const send = () => {
    fetch('/api/group/' + id, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        has_errors: flag.checked,
        comment: comment.value,
      }),
    }).then(r => {
      if (r.ok) {
        div.classList.toggle('has-errors', flag.checked);
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


def create_app(
    *,
    groups_json: Path = GROUPS_JSON,
    photos_dir: Path = RAW_PHOTOS_DIR,
) -> Flask:
    """Build the review Flask app.

    ``photos_dir`` is the *fallback* lookup directory for thumbnails — the
    actual lookup first tries to resolve the path stored in
    ``groups.json`` (which may live in a sample directory, in
    ``data/raw_photos/``, or anywhere). Photos arriving via the
    extension-pool boundary-resolution stage carry repo-relative paths,
    so we have to honour those too.
    """
    app = Flask(__name__)

    def resolve_photo(filename: str) -> Path | None:
        """Find the on-disk JPEG for ``filename`` (a basename).

        Looks first at every photo path stored in groups.json (resolved
        against the groups_json file's directory for repo-relative
        entries), then falls back to ``photos_dir / filename``.
        """
        groups = _load_groups(groups_json)
        groups_root = groups_json.resolve().parent
        for g in groups:
            photos = g["photos"]
            assert isinstance(photos, list)
            for p in photos:
                stored = Path(str(p["path"]))
                if stored.name != filename:
                    continue
                candidate = (
                    stored if stored.is_absolute() else (groups_root / stored)
                )
                if candidate.exists():
                    return candidate
                # The plan stores paths repo-relative to the project root
                # (data/raw_photos/...). Try that too.
                repo_relative = REPO_ROOT / stored
                if repo_relative.exists():
                    return repo_relative
        fallback = photos_dir / filename
        return fallback if fallback.exists() else None

    @app.get("/")
    def index() -> str:
        groups = _load_groups(groups_json)
        for g in groups:
            photos = g["photos"]
            assert isinstance(photos, list)
            for p in photos:
                p["filename"] = Path(p["path"]).name
            g.setdefault("has_errors", False)
            g.setdefault("comment", "")
        return render_template_string(INDEX_TEMPLATE, groups=groups)

    @app.get("/thumb/<path:filename>")
    def thumb(filename: str) -> Response:
        src = resolve_photo(filename)
        if src is None:
            abort(404)
        return _serve_thumbnail(src)

    @app.get("/raw/<path:filename>")
    def raw(filename: str) -> Response:
        src = resolve_photo(filename)
        if src is None:
            abort(404)
        return send_file(str(src.resolve()), mimetype="image/jpeg")

    @app.post("/api/group/<group_id>")
    def update_group(group_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        has_errors = bool(payload.get("has_errors", False))
        comment = str(payload.get("comment", ""))
        if not _update_group_in_place(
            groups_json, group_id, has_errors, comment
        ):
            abort(404)
        return jsonify({"ok": True})

    return app


def _load_groups(groups_json: Path) -> list[dict[str, Any]]:
    if not groups_json.exists():
        return []
    payload = json.loads(groups_json.read_text())
    groups: list[dict[str, Any]] = payload.get("groups", [])
    return groups


def _update_group_in_place(
    groups_json: Path, group_id: str, has_errors: bool, comment: str
) -> bool:
    payload = json.loads(groups_json.read_text())
    for g in payload.get("groups", []):
        if g.get("id") == group_id:
            g["has_errors"] = has_errors
            g["comment"] = comment
            groups_json.write_text(json.dumps(payload, indent=2))
            return True
    return False


def _serve_thumbnail(src: Path) -> Response:
    with Image.open(src) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented is None:  # pragma: no cover
            raise RuntimeError(f"exif_transpose returned None for {src}")
        oriented.thumbnail(
            (THUMBNAIL_LONGEST_SIDE, THUMBNAIL_LONGEST_SIDE),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        oriented.convert("RGB").save(buf, "JPEG", quality=80)
        buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review server for groups.json"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--groups-json", type=Path, default=GROUPS_JSON)
    parser.add_argument("--photos-dir", type=Path, default=RAW_PHOTOS_DIR)
    args = parser.parse_args()

    app = create_app(groups_json=args.groups_json, photos_dir=args.photos_dir)
    print(f"Review server: http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
