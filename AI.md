# AI.md — mmappet (Rust)

Agent instructions for this repo. `CLAUDE.md`, `AGENTS.md`, `GEMINI.md` and
`.github/copilot-instructions.md` are symlinks to this file; edit only this one.

## What this is

The Rust copy of the Python `mmappet` package (`github.com/michalsta/mmappet`): the same
directory format (`schema.txt` with one `<numpy dtype> <name>` line per column, column `i`
in `i.bin`, native byte order, no header), read and written the way `mmappet.py` does.
The C++ header in that repo is a third implementation of the same format.

## Python is the oracle

When the Rust and Python behaviour differ, Python is right, unless the difference is a
deliberate refusal listed here. The oracle tests (`tests/oracle.rs`) run
`tests/python/oracle.py` in `.venv`, which `make venv` builds with `uv` (Python 3.12,
downloaded if missing) and Python mmappet pinned by commit in the `Makefile`. They call
`make venv` themselves, so `cargo test` needs only `uv` and `git` on PATH. Bump the pin
deliberately; the checks compare bytes, so a new Python version can change what passes.

Deliberate refusals:

- Columns of unequal length are an error on read (Python's dict reader does not check).
- Time columns hold only `s`, `ms`, `us` or `ns` (`TimeUnit`). Python builds its schema
  through pandas, which reads any other numpy unit under a wrong label (`datetime64[D]`
  ticks come back as seconds) and refuses multiples such as `10ms`; Rust refuses them all.
  Python's `DatasetWriter.append` cannot write time columns at all; `append_df` can.

## Features

The library has no default features. `cli` (the `mmappet-cli` binary) pulls in `clap`,
`anyhow` and `ndarray`; `ndarray` alone adds the `ArrayView1` accessors. Keep `clap` and
`ndarray` out of the default build: downstream users (SAGE) take only the library.

## Formatting

Every `.rs` file is rustfmt-clean. Agents: run `rustfmt --edition 2021 <file>` after every
write or edit of a `.rs` file, as a post-edit hook where the agent has hooks (Claude Code:
`PostToolUse` on `Write|Edit`), else by hand. `make hooks` points git at `githooks/`, whose
`pre-commit` formats and re-stages staged `.rs` files and refuses partly staged ones.

## Docs

Present tense, only what the code cannot say, about 60 lines; history goes in commit
messages. Ask before adding a doc file.
