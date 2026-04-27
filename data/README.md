# `data/`

Pipeline data lives here. Most subdirectories are gitignored (raw photos, stitched intermediates, the LLM cache); the small JSON artifacts that we want as provenance / regression fixtures are committed.

## `gold_groups.json`

The 208-photo session from 2026-04-26 (Mom's Organic Market and CVS), grouped by product and reviewed by hand.

Process:

1. `scripts/group_photos.py` was iterated to convergence on a 45-photo kinks-first sample.
1. The full 208 photos were grouped with the converged prompt and reviewed in `scripts/review_server.py`.
1. The reviewer flagged zero grouping/role errors; eight groups carried a `comment` noting that a particular wrap-around photo was *missing* a `nutrition` role label that a strict gold standard would include. Per the relaxed `nutrition` semantics those omissions are not "errors" for the live pipeline, but the gold standard adds the role back so this file is suitable as a reference set for future prompt iterations.
1. `gold_groups.json` is the output of step 2 with those eight role corrections applied. Comments are preserved so a future audit can see exactly which photos differed and why.

The format mirrors `data/groups.json` (the live pipeline output): one top-level `groups` array, each entry with `id`, `store`, `photos: [{path, roles}]`, `warnings`. Review-UI metadata (`has_errors`, `locked`, empty `comment`) has been stripped.

Use this file as the regression target whenever the grouping prompt changes — re-run `group_photos.py` against `data/raw_photos/` and diff against `gold_groups.json`. Any new disagreement is a candidate failure mode worth investigating.
