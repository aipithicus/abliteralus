# Supply-chain policy

The committed `uv.lock` is the reproducible dependency source for the supported
Python 3.10–3.12 test matrix. CI installs CPU-only PyTorch from PyTorch's
explicit package index and resolves every other package from PyPI. The lock
contains exact versions, source URLs, environment markers, and artifact hashes.

CI uses exact Python tool versions and checksum-pinned standalone binaries.
`ci/digests.txt` records executable and GitHub Action pins; build, test, and
license-tool transitive dependencies are covered by `uv.lock`.

## Digest-bound release evidence

The release artifact is a deterministic ZIP snapshot of the tested repository
commit. No wheel, source distribution, executable, or other compiled package is
published. The release job creates the snapshot once with `git archive`, then
emits a bound CycloneDX source SBOM and a canonical `SHA256SUMS` manifest.

GitHub's keyless Sigstore-backed attestation service signs two in-toto
statements for that release set:

- SLSA build provenance for every subject named by `SHA256SUMS`;
- an SBOM attestation binding the CycloneDX document to the source ZIP digest.

The supply-chain job downloads the package job's retained artifact, verifies
`SHA256SUMS`, and revalidates the SBOM binding. It must not create a substitute
archive. Publication must use this same tested and attested ZIP. A commit,
dependency lock, snapshot instruction, artifact, checksum manifest, SBOM, or
release-policy change invalidates the evidence.

Consumers can perform the portable integrity check with:

```bash
sha256sum --check SHA256SUMS
```

For authenticity and provenance, verify the source ZIP against the canonical
repository with GitHub's attestation verifier. A checksum without its signed
provenance proves byte integrity only; it does not prove who built the artifact
or which source and build instructions produced it.

## Required evidence

The Supply chain job retains these artifacts for 14 days:

- one redacted Gitleaks JSON report for the checkout;
- OSV audit JSON and scanner status for Python 3.10, 3.11, and 3.12 on Linux;
- a JSON license inventory for all supported dependency extras;
- a CycloneDX 1.5 source SBOM bound to the repository ZIP by SHA-256;
- the policy decisions and exact tested source ZIP used by those checks.

Every secret finding and every OSV vulnerability is blocking by default. OSV
does not provide a normalized severity for every advisory, so the policy treats
unknown, low, medium, high, and critical findings alike. This is stricter than
a high-only threshold and avoids silently passing advisories with missing
severity data.

License metadata must exactly match an expression in
`ci/supply-chain-policy.json`. ABLITERALUS itself is excluded from dependency
license evaluation because its AGPL license is the project license rather than
a third-party dependency decision.

## Exceptions

Exceptions live only in `ci/supply-chain-policy.json`; command-line ignores and
unconditional success conversion are forbidden.

- A vulnerability exception names an OSV/GHSA/CVE identifier, states a reason,
  records `approved_on` and `expires`, and declares whether a fix is available.
  Fixable findings may be excepted for at most 7 days; findings without a fix
  may be excepted for at most 90 days.
- A secret exception names the redacted Gitleaks fingerprint, states a reason,
  and records `approved_on` and `expires`. It may last at most 30 days.
- Expired, overlong, stale, malformed, or unused exceptions fail CI. A finding
  that gains a fix cannot use an exception declared as unfixable.
- License exceptions are not supported. Add an exact expression only after a
  maintainer verifies the package metadata and compatibility with AGPL-3.0-or-later.

## Updating the lock and tools

Use the version of uv recorded in `ci/digests.txt`:

```bash
uvx --from uv==0.12.4 uv lock --upgrade
uvx --from uv==0.12.4 uv lock --check
```

Review the complete lock diff, source indexes, new licenses, vulnerability
evidence, and SBOM diff. Update direct pins in `pyproject.toml` and executable
pins/checksums in `ci/digests.txt` in the same pull request. A normal dependency
or tool update must not add an exception merely to make CI green.
