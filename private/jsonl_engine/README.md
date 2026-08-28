# JSONL engine

`aipithicus-jsonl-engine` provides deterministic JSONL framing and transactional, in-place
append for ABLITERALUS-owned stores. It has no knowledge of research journals, prompt batteries,
models, experiments, or workspace discovery.

`JsonlStore` is the single physical-store owner. It normalizes the data path and owns the adjacent
lease, transaction table, JIDX, canonical prefix inspection, durable backup and suffix repair, and
derived-state rebuild. Domain packages compose this object instead of implementing another storage
layer.

Each logical store has four stable paths:

- the JSONL data file;
- one binary JSOI v2 `{stem}.jidx` byte-offset index;
- one append-only `<store>.transactions.jsonl` commit table;
- one `<store>.lock` cross-process write lease.

A transaction serializes and validates its records before mutation, appends the batch under the
lease, flushes the data, incrementally appends the new record offsets to JIDX, advances its fixed
header, and then appends one commit row. The commit row records the generation, committed byte
boundary, record count, cumulative data and JIDX hashes, an optional caller-owned partition
descriptor, and caller-owned typed metadata. Ordinary appends read only the last commit row and the
fixed JIDX header; neither the data file nor the existing offset table is rebuilt.

JSOI v2 preserves the historical `JSOI` magic, version, record count, source length, source
last-write ticks, and one little-endian int64 offset per record. `read_record` and `read_range` use
those offsets for bounded seeks. `record_count` is constant-time. `partition_ranges` and
`read_partition` map generic nested descriptors such as tier and factorial-cell coordinates onto
committed batch ranges without teaching the engine prompt-domain semantics.

JIDX is derived but essential: every transaction binds its expected JIDX length and offset-chain
hash. Missing, interrupted, or stale index generations are rebuilt deterministically from the
authoritative committed JSONL bytes. Explicit `verify` compares every indexed offset with a full
data scan; `rebuild_index` provides the maintenance operation directly.

The transaction table is not a shard collection or a per-record projection. One row commits a batch
of any bounded size. A crash before the commit row leaves data or JIDX bytes beyond the last
committed boundary; the generic engine rolls them back or rebuilds derived index state by default,
while clients that require preview-first data repair can select fail-closed recovery.

The first client is `aipithicus-research-journal`. Its record schema, privacy policy, hash chain,
queries, and the decision to apply a previewed repair remain in that package; physical repair is a
`JsonlStore` operation.
