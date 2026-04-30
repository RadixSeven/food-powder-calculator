# Overview

This is a utility for handing powder mixing for my own personal use. It
provided a chance to practice using the `pyomo` library and deal with the
yak shaving it requires.

The following instructions are intended for my future self.

# Development Installation

```shell
git lfs install                  # one-time per machine: enable LFS hooks
uv sync                          # populate .venv from pyproject.toml
uv run pre-commit install        # install the git pre-commit hook
```

`uv sync` is also implicitly run by every `uv run ...` invocation below, so a
fresh checkout can skip straight to the commands in the next sections — but
`pre-commit install` still has to be run once per clone to enable the
git hook.

`git lfs install` is required because the raw product photos under
`data/raw_photos/` are stored via Git LFS. Without it, a `git pull` would
leave LFS pointer files in place of the actual images. If you've cloned
without running it, run `git lfs install && git lfs pull` to fetch the
photos retroactively.

# Running

```shell
pants run src:main # I had a "better" name but this was easier to remember
```

# Testing and static analysis

```shell
./run_all_qa.sh               # convenience: runs every gate below in order
```

Or run them individually:

```shell
uv run ruff format .          # format Python
uv run mdformat .             # format Markdown
uv run ruff check --fix .     # lint Python and auto-fix
uv run pyrefly check          # strict type-check (config under [tool.pyrefly] in pyproject.toml)
pants lint ::                 # ruff via pants
uvx --from shellcheck-py shellcheck --severity=style --enable=all $(git ls-files '*.sh')
pants test ::
```

Pyrefly is the project's type checker. The config in `pyproject.toml` enables
its strict mode (`check-unannotated-defs`, `strict-callable-subtyping`,
`spec-compliant-overloads`, `permissive-ignores=false`, and explicit error
promotion for `implicit-any`, `unannotated-*`, `redundant-*`, and
`unused-ignore`). Pants no longer runs mypy.

The `uv run` invocations auto-sync `.venv` so that pyrefly can resolve
imports of `pyomo`, `highspy`, `pytest`, etc. — no manual `pip install` step
is needed.

Before committing, all tests and static analysis must pass, tests must have
100% coverage, and the tree must be fully formatted.

# Reproducibility: durable artifacts vs cache

The pipeline distinguishes two kinds of on-disk state, and the
deletion-survival rule is the linter for the boundary:

| Where | Tracked? | Role |
| ----------------- | -------- | ------------------------------------------------------------------------------- |
| `data/raw_photos/<batch>/` | yes (LFS) | Original images, organized per batch with a `batch.yaml`. |
| `data/llm_records/<sha>.json` | yes | Every (request, response) pair the pipeline has ever observed. |
| `data/runs/<id>/manifest.yaml` | yes | Per-invocation audit record (git sha, argv, inputs/outputs, calls). |
| `data/extracted_yaml/<gid>.yaml` | yes | Final per-group nutrition data the optimizer consumes. |
| `data/gold_groups.json` | yes | Hand-locked photo→group assignment. |
| `data/cache/` | **no** | Performance cache. Deletable; rebuilt from durable records on demand. |
| `data/stitched_panels/` | **no** | Per-group bbox crops. Deletable; rebuilt by `group_pipeline.py`. |

**The rule:** `rm -rf data/cache/` followed by re-running any pipeline
command must produce the same outputs without a single LLM call. If
that fails, the cache became load-bearing — it stopped being a cache.
Code MUST NOT take cache paths as arguments crossing command
boundaries; the cache is private memoization inside `_claude.call` and
`crop_panel`.

# Adding a new shopping batch

```shell
# 1. Drop the photos under data/raw_photos/<batch-name>/
mkdir -p data/raw_photos/2026-08-15-vitamins
cp ~/Downloads/PXL_*.jpg data/raw_photos/2026-08-15-vitamins/

# 2. Write batch.yaml describing the batch
cat > data/raw_photos/2026-08-15-vitamins/batch.yaml <<'YAML'
name: 2026-08-15-vitamins
captured_at: "2026-08-15"
source: phone-camera
notes: |
  Whole-foods supplement aisle, single store.
stores:
  - {name: WholeFoods}
YAML
```

The pipeline then picks up the batch automatically — no code change
needed to map photos to stores or to recognize them as part of the
session. Cross-batch listing in `group_photos.list_photos` sorts by
filename so PXL timestamps still order correctly.

# Pipeline commands

Each pipeline-step command writes a `data/runs/<id>/manifest.yaml`
recording the inputs (with sha256), outputs (with sha256), git sha,
argv, and every LLM call it made. The run id is sortable
(`<ISO-second>-<short-git-sha>`), with a trailing counter on
collisions.

```shell
# Bbox + crop + stitch for one or more groups.
uv run python scripts/group_pipeline.py 20260426_mom_001 20260426_mom_002

# Extract per-group YAML (consumes the manifest from group_pipeline).
uv run python scripts/extract_yaml.py 20260426_mom_001
```

If a step's declared input doesn't exist, the wrapper raises with a
message pointing at the command that produces it:

```
[extract_yaml.process_group_to_yaml] missing input: data/gold_groups.json
  produce it with: uv run python scripts/group_pipeline.py <gid>
```

# Provenance lookup

When something looks suspicious or unexpected:

```shell
uv run python scripts/provenance.py data/extracted_yaml/20260426_mom_001.yaml
```

prints the run that produced it (most recent first), with the
matching output's recorded sha256 and the producing command's argv.
Exit 1 on no match so shell pipelines can branch on it.

# Cache deletion (the linter for the boundary)

```shell
rm -rf data/cache/
uv run python scripts/group_pipeline.py 20260426_mom_001  # re-runs from records
```

Should produce identical outputs in `data/stitched_panels/<gid>/` and
fill the cache back in. If a command fails because a cache file is
missing, the cache became load-bearing somewhere — fix the boundary,
don't add a "make sure cache is present" step.

# Backfilling records from a legacy cache

The durable-records layer was added partway through the project; old
entries that exist only in `data/cache/responses/` can be promoted
once with:

```shell
uv run python scripts/backfill_llm_records.py
```

Idempotent — re-runs skip records that already exist.
