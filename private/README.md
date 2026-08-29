# ABLITERALUS private lab

`private/` is the boundary for Aipithicus-owned research, experiment, and
operations code that is not part of the OBLITERATUS contribution shadow. The
repository root remains the renamed public fork plus deliberately upstreamable
changes. Private modules may consume public ABLITERALUS contracts; public
modules must not import or discover private implementations.

## Layout

- `src/` contains the private Python import packages.
- `tests/` mirrors those packages without introducing separate project roots.
- `config/` contains tracked examples and ignored machine-local configuration.
- `experiments/` contains lab-owned experiment definitions and prompt datasets.
- `docs/` contains substantive operator and component documentation.
- `tools/shadow/` owns fork synchronization and write-back tooling.
- `contributions/pr-drafts/` contains generated upstream PR drafts.

The private packages remain separate domain owners even though they are shipped
and tested as one repository-local distribution:

```text
research_journal -> jsonl_engine

lab_bench -> research_journal
          -> secret_manager
          -> surgery_artifacts
          -> public abliteralus contracts

surgery_artifacts -> public abliteralus contracts
guard_study       -> public abliteralus analysis and model APIs
```

Future private capabilities should normally be added directly beneath `src/`
and `tests/`. A new nested project and `pyproject.toml` require an actual
independent release or deployment boundary, not merely a new module.

## Environment

The private project owns `private/.venv` and `private/uv.lock`. The root
environment remains reserved for the public shadow and contribution gates.
Restore the repository-owned uv executable, then synchronize the private
environment from the repository root:

```powershell
pwsh -NoProfile -File deps/uv/restore-uv.ps1
pwsh -NoProfile -File deps/uv/run-uv.ps1 sync --project private --group dev
```

Copy the tracked configuration examples before using the operator tools:

```text
private/config/lab.example.toml     -> private/config/lab.local.toml
private/config/secrets.example.toml -> private/config/secrets.local.toml
```

Run managed tests through the private lab bench so scratch, cache, coverage, and
temporary state remain repository-local:

```text
private/.venv/Scripts/lab-bench.exe --config private/config/lab.local.toml test --cwd private -- -q
```

The same environment provides `abliteralus-lightning`, `guard-study`,
`lab-bench`, `research-journal`, `secret-broker`, `secret-manager`, and
`surgery-artifact`.

## Documentation

- [Lab bench](docs/lab-bench.md)
- [Surgery experiment bench](docs/surgery-experiment-bench.md)
- [Guard study](docs/guard-study.md)
- [JSONL engine](docs/jsonl-engine.md)
- [Research journal](docs/research-journal.md)
- [Secret manager](docs/secret-manager.md)
- [Surgery artifacts](docs/surgery-artifacts.md)
- [Shadow synchronization and write-back](docs/shadow-sync.md)
