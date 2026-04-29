"""Entry point for the multi-view review server.

The server hosts one view per pipeline stage, each in its own module:

* ``/groups/`` — group-and-role review (:mod:`_review_groups`).
* ``/crops/``  — bbox/crop review (:mod:`_review_crops`).

A view is enabled when its inputs are reachable: the groups view always
runs (the ``--groups-json`` default exists by Phase A); the crops view
only registers when the gold-groups JSON and the per-group stitched
output root both exist, since it needs the per-group manifests written
by :mod:`group_pipeline`.

Run:

    uv run python scripts/review_server.py [--port 8765]

Then open http://localhost:8765/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flask import Flask, redirect, render_template_string
from werkzeug.wrappers.response import Response as WerkzeugResponse

import _review_crops
import _review_groups
from group_photos import GROUPS_JSON, RAW_PHOTOS_DIR
from group_pipeline import GOLD_GROUPS_JSON, STITCHED_PANELS_DIR

LANDING_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Review server</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 40em; margin: 2em auto; }
ul { line-height: 1.8; }
.disabled { color: #888; }
.disabled .why { font-size: 0.85em; color: #c53030; }
</style></head><body>
<h1>Review server</h1>
<p>Pick a stage to review. Each link opens an independent page; you can have
both open in different tabs.</p>
<ul>
{% for view in views %}
  {% if view.enabled %}
    <li><a href="/{{ view.prefix }}/">{{ view.label }}</a></li>
  {% else %}
    <li class="disabled">{{ view.label }}
      <span class="why">(disabled: {{ view.why_disabled }})</span></li>
  {% endif %}
{% endfor %}
</ul>
</body></html>
"""


def create_app(
    *,
    groups_json: Path = GROUPS_JSON,
    photos_dir: Path = RAW_PHOTOS_DIR,
    gold_groups_json: Path | None = GOLD_GROUPS_JSON,
    stitched_root: Path | None = STITCHED_PANELS_DIR,
    crop_reviews_json: Path = _review_crops.DEFAULT_CROP_REVIEWS_JSON,
) -> Flask:
    """Build the review Flask app.

    The groups view registers unconditionally (its inputs are always
    available once Phase A has run). The crops view registers only when
    both ``gold_groups_json`` and ``stitched_root`` are non-None — this
    is what tests use to spin up a groups-only or crops-only fixture
    without having to fake the other view's inputs.
    """
    app = Flask(__name__)
    _review_groups.register(app, groups_json=groups_json, photos_dir=photos_dir)
    crops_enabled = gold_groups_json is not None and stitched_root is not None
    if (
        crops_enabled
        and gold_groups_json is not None
        and stitched_root is not None
    ):
        _review_crops.register(
            app,
            gold_groups_json=gold_groups_json,
            stitched_root=stitched_root,
            crop_reviews_json=crop_reviews_json,
        )

    @app.get("/")
    def landing() -> str:
        views = [
            {
                "prefix": "groups",
                "label": "Groups & roles",
                "enabled": True,
                "why_disabled": "",
            },
            {
                "prefix": "crops",
                "label": "Crops",
                "enabled": crops_enabled,
                "why_disabled": (
                    "no gold-groups JSON or stitched-panels directory configured"
                ),
            },
        ]
        return render_template_string(LANDING_TEMPLATE, views=views)

    @app.get("/groups")
    def groups_redirect() -> WerkzeugResponse:
        # Trailing-slash redirect so /groups (without slash) still works.
        return redirect("/groups/", code=302)

    @app.get("/crops")
    def crops_redirect() -> WerkzeugResponse:
        return redirect("/crops/", code=302)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-view review server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--groups-json",
        type=Path,
        default=GROUPS_JSON,
        help="Groups review state (data/groups.json by default).",
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=RAW_PHOTOS_DIR,
        help="Fallback raw-photos lookup directory.",
    )
    parser.add_argument(
        "--gold-groups-json",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Locked groups for crop review (data/gold_groups.json).",
    )
    parser.add_argument(
        "--stitched-root",
        type=Path,
        default=STITCHED_PANELS_DIR,
        help="Per-group stitched-panel output root (where manifests live).",
    )
    parser.add_argument(
        "--crop-reviews-json",
        type=Path,
        default=_review_crops.DEFAULT_CROP_REVIEWS_JSON,
        help="Crop review state (data/crop_reviews.json by default).",
    )
    args = parser.parse_args()

    app = create_app(
        groups_json=args.groups_json,
        photos_dir=args.photos_dir,
        gold_groups_json=args.gold_groups_json,
        stitched_root=args.stitched_root,
        crop_reviews_json=args.crop_reviews_json,
    )
    print(f"Review server: http://{args.host}:{args.port}/", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
