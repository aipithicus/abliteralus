# Surgery artifacts

`surgery_artifacts` stores an ABLITERALUS surgery as an exact, base-anchored
capsule instead of treating a complete copied checkpoint as the durable unit.

The native capsule is the source of truth. PEFT and llama.cpp adapters are
derived exports and are emitted only when the native operations can be
represented by the target format.

The package is private infrastructure. Public ABLITERALUS code exposes a
neutral artifact-stage callback and does not import this package.

## Capsule v1

A capsule directory contains:

- `manifest.json`: strict capsule identity, base identity, ordered operations,
  and safe ancillary-file overlays.
- `tensor-manifest.json`: base tensor hashes and expected hashes for every
  changed tensor.
- `delta.safetensors`: non-executable tensor payloads.
- `recipe.json`: experiment and runtime provenance.
- `files/`: safe, non-weight files copied from the surgery checkpoint.
- `SHA256SUMS`: integrity hashes for every capsule file except itself.

The surgery id is derived from the base tensor identity plus the exact identities
of every changed target tensor. It deliberately excludes timestamps, compression,
the selected storage codec, and derived exports.

## Commands

```text
surgery-artifact create --base BASE --target TARGET --output CAPSULE
surgery-artifact verify CAPSULE
surgery-artifact rehydrate --capsule CAPSULE --base BASE --output CHECKPOINT
surgery-artifact registry add --registry REGISTRY --capsule CAPSULE --ref NAME
surgery-artifact export-peft --capsule CAPSULE --output ADAPTER
```

All commands fail closed. Pickle files and executable checkpoint payloads are
never copied into a capsule.
