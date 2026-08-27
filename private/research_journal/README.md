# Research journal

`aipithicus-research-journal` is a small, application-neutral JSONL journal core. It records
immutable research entries without importing ABLITERALUS, Lightning, Hugging Face, or any sharing
backend. The lab bench supplies those domain relations at its own boundary.

The store uses compact canonical UTF-8 JSON, one LF-terminated object per line, a cross-process
write lease, durable append (`flush`/`fsync` semantics), UUID journal and entry identities, and a
SHA-256 record chain. Verification is strict: malformed, non-canonical, unterminated, reordered, or
hash-invalid records make later append fail closed. `repair` is preview-first; applying it preserves
the complete original as a timestamped `.bak` before removing the invalid suffix.

The first line is a `research-journal/header`; later lines are `research-journal/entry` records.
The portable JSON Schema is shipped as `schemas/journal-record-v1.schema.json`.

## Privacy and sharing

Entries default to `sharing=private`. `candidate` and `approved` are review states only: this package
never uploads, publishes, or exports anything. It rejects common credential fields, token formats,
and private-key material, and it never captures commands or environment variables automatically.
The filter is a safety guard, not a substitute for reviewing material before publication.

OBLITERATUS already has explicit community-contribution JSON and opt-in aggregate telemetry. A future
adapter should project deliberately approved journal result entries into that existing public schema;
the private journal should not become another telemetry channel.

## Standalone use

```text
research-journal --journal outputs/research-journal/journal.jsonl add \
  --kind hypothesis --title "Guard steering should raise the unsafe margin" \
  --tag llama-guard --relation run=guard-sweep-001

research-journal --journal outputs/research-journal/journal.jsonl list --limit 20
research-journal --journal outputs/research-journal/journal.jsonl verify
research-journal --journal outputs/research-journal/journal.jsonl repair
research-journal --journal outputs/research-journal/journal.jsonl repair --apply
```

Use `--body-file -` or `--data-file -` for stdin. The two cannot share stdin in one invocation.
