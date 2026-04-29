"""Shared infrastructure for the review-server views.

Each view (groups, crops, future YAML) is a self-contained module that
calls :func:`register_image_routes` once for its URL prefix and then
adds its own pages and POST handlers. Putting the image-serving helpers
here keeps every view's thumbnail/raw behaviour identical (same
LANCZOS resampling, same EXIF-transpose) without copy/paste.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable

from flask import Flask, Response, abort, send_file
from PIL import Image, ImageOps

from _json_types import JsonObject, JsonValue

THUMBNAIL_LONGEST_SIDE = 320


def serve_thumbnail(src: Path) -> Response:
    """EXIF-orient the source, downscale to a thumbnail, return JPEG bytes.

    Used by every view's ``/<prefix>/thumb/...`` route. Returning a
    fresh BytesIO on each call (rather than caching) is fine: the
    review UI is one user and the source images are SSD-local.
    """
    with Image.open(src) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented is None:  # pragma: no cover — only None for None input
            raise RuntimeError(f"exif_transpose returned None for {src}")
        oriented.thumbnail(
            (THUMBNAIL_LONGEST_SIDE, THUMBNAIL_LONGEST_SIDE),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        oriented.convert("RGB").save(buf, "JPEG", quality=80)
        buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


def serve_raw(src: Path) -> Response:
    """Return the on-disk JPEG bytes unchanged."""
    return send_file(str(src.resolve()), mimetype="image/jpeg")


def register_image_routes(
    app: Flask,
    *,
    url_prefix: str,
    resolver: Callable[[str], Path | None],
    endpoint_prefix: str,
) -> None:
    """Wire ``/{url_prefix}/thumb/<key>`` and ``/{url_prefix}/raw/<key>``.

    ``resolver`` maps the URL path key to an on-disk path or ``None``;
    each view encodes whatever lookup logic it needs (groups: just the
    filename; crops: ``<group_id>/<role>/<filename>``).

    Endpoint names use ``endpoint_prefix`` so multiple views can each
    register thumb/raw routes without colliding in Flask's URL map.
    """

    def thumb(key: str) -> Response:
        src = resolver(key)
        if src is None:
            abort(404)
        return serve_thumbnail(src)

    def raw(key: str) -> Response:
        src = resolver(key)
        if src is None:
            abort(404)
        return serve_raw(src)

    app.add_url_rule(
        f"/{url_prefix}/thumb/<path:key>",
        endpoint=f"{endpoint_prefix}_thumb",
        view_func=thumb,
        methods=["GET"],
    )
    app.add_url_rule(
        f"/{url_prefix}/raw/<path:key>",
        endpoint=f"{endpoint_prefix}_raw",
        view_func=raw,
        methods=["GET"],
    )


def load_json_object(path: Path) -> JsonObject | None:
    """Read a JSON file expected to have an object at the top level.

    Returns the parsed dict on success, ``None`` if the file is
    missing or its contents are not a JSON object. Used by every view
    that persists state to disk; the silent-None return lets the
    caller render an empty UI instead of crashing on a fresh repo.
    """
    if not path.exists():
        return None
    payload: JsonValue = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload
