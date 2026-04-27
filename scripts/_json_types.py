"""Recursive type aliases for JSON values.

JSON has a precise type structure: each value is None, bool, int, float,
str, an array of values, or an object (str-keyed dict of values). Using
``dict[str, Any]`` to model that is lossy — every nested access is silently
typed ``Any`` and the type system stops helping. The recursive aliases here
let pyrefly track the structure end-to-end; callers narrow with
``isinstance`` at the points they need a concrete shape.
"""

from __future__ import annotations

type JsonValue = None | bool | int | float | str | list[JsonValue] | JsonObject
type JsonObject = dict[str, JsonValue]
