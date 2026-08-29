"""Domain-owned reconstruction over the generic transactional JSONL store."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jsonl_engine import JsonlStore

from .contracts import EVIDENCE_KINDS, make_event, new_id, validate_event
from .errors import ProbeHarnessError


@dataclass(slots=True)
class TrialView:
    """One completed trial plus annotations folded from later events."""

    trial_id: str
    payload: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    marked: bool = False
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    session_id: str
    subject: dict[str, Any]
    settings: dict[str, Any]
    trials: dict[str, TrialView]
    failed_trials: dict[str, dict[str, Any]]
    observations: dict[str, dict[str, Any]]
    observation_order: list[str]
    direction_bundles: list[dict[str, Any]]

    @property
    def latest_trial(self) -> TrialView | None:
        return next(reversed(self.trials.values()), None) if self.trials else None

    def resolve_trial(self, value: str | None = None) -> TrialView:
        if value is None:
            trial = self.latest_trial
            if trial is None:
                raise ProbeHarnessError("the session has no completed trials")
            return trial
        if value in self.trials:
            return self.trials[value]
        matches = [trial for trial_id, trial in self.trials.items() if trial_id.startswith(value)]
        if not matches:
            raise ProbeHarnessError(f"unknown trial: {value}")
        if len(matches) > 1:
            raise ProbeHarnessError(f"ambiguous trial prefix: {value}")
        return matches[0]

    @property
    def latest_observation(self) -> dict[str, Any] | None:
        if not self.observation_order:
            return None
        return self.observations[self.observation_order[-1]]

    def resolve_observation(
        self,
        value: str | None = None,
        *,
        evidence_kind: str | None = None,
    ) -> dict[str, Any]:
        candidates = [
            self.observations[observation_id]
            for observation_id in self.observation_order
            if evidence_kind is None
            or self.observations[observation_id].get("evidence_kind") == evidence_kind
        ]
        if value is None:
            if not candidates:
                qualifier = f" {evidence_kind}" if evidence_kind else ""
                raise ProbeHarnessError(f"the session has no{qualifier} observations")
            return candidates[-1]
        exact = self.observations.get(value)
        if exact is not None:
            if evidence_kind is not None and exact.get("evidence_kind") != evidence_kind:
                raise ProbeHarnessError(
                    f"observation {value} is not evidence kind {evidence_kind}"
                )
            return exact
        matches = [
            observation
            for observation in candidates
            if str(observation["observation_id"]).startswith(value)
        ]
        if not matches:
            raise ProbeHarnessError(f"unknown observation: {value}")
        if len(matches) > 1:
            raise ProbeHarnessError(f"ambiguous observation prefix: {value}")
        return matches[0]


class SessionStore:
    """Append events and reconstruct one interactive probing session."""

    def __init__(self, directory: str | Path, *, session_id: str) -> None:
        self.directory = Path(directory).resolve()
        self.session_id = session_id
        self.events = JsonlStore(self.directory / "events.jsonl")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        subject: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> "SessionStore":
        root_path = Path(root).resolve()
        session_id = new_id("session")
        directory = root_path / session_id
        directory.mkdir(parents=True, exist_ok=False)
        session = cls(directory, session_id=session_id)
        session._append(
            "session.started",
            {
                "subject": dict(subject),
                "settings": dict(settings),
            },
        )
        return session

    @classmethod
    def resume(
        cls,
        directory: str | Path,
        *,
        expected_subject_digest: str | None = None,
    ) -> "SessionStore":
        resolved = Path(directory).resolve()
        events = JsonlStore(resolved / "events.jsonl")
        records = events.read_records()
        if not records:
            raise ProbeHarnessError(f"probe session has no committed events: {resolved}")
        first = records[0]
        validate_event(first)
        if first["event_type"] != "session.started":
            raise ProbeHarnessError("the first probe-session event is not session.started")
        session_id = str(first["session_id"])
        session = cls(resolved, session_id=session_id)
        state = session.state(records=records)
        actual_digest = state.subject.get("digest")
        if expected_subject_digest is not None and actual_digest != expected_subject_digest:
            raise ProbeHarnessError(
                "the requested subject does not match the subject recorded by the session"
            )
        return session

    def start_trial(
        self,
        *,
        prompt: str,
        parent_trial_id: str | None,
        replay_of: str | None,
        generation: Mapping[str, Any],
    ) -> str:
        trial_id = new_id("trial")
        self._append(
            "trial.started",
            {
                "trial_id": trial_id,
                "parent_trial_id": parent_trial_id,
                "replay_of": replay_of,
                "prompt": prompt,
                "generation": dict(generation),
            },
        )
        return trial_id

    def complete_trial(self, trial_id: str, payload: Mapping[str, Any]) -> None:
        if payload.get("trial_id") != trial_id:
            raise ProbeHarnessError("completed trial payload does not match its trial ID")
        self._append("trial.completed", payload)

    def fail_trial(self, trial_id: str, error: BaseException) -> None:
        self._append(
            "trial.failed",
            {
                "trial_id": trial_id,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )

    def annotate(
        self,
        trial_id: str,
        *,
        note: str | None = None,
        tag: str | None = None,
        marked: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {"trial_id": trial_id}
        if note is not None:
            value = note.strip()
            if not value:
                raise ProbeHarnessError("note must not be empty")
            payload["note"] = value
        if tag is not None:
            value = tag.strip()
            if not value or any(character.isspace() for character in value):
                raise ProbeHarnessError("tag must be one non-empty token")
            payload["tag"] = value
        if marked is not None:
            payload["marked"] = bool(marked)
        if len(payload) == 1:
            raise ProbeHarnessError("annotation contains no change")
        self._append("trial.annotated", payload)

    def record_observation(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload.get("trial_id"), str):
            raise ProbeHarnessError("observation must reference a trial")
        if not isinstance(payload.get("observation_id"), str):
            raise ProbeHarnessError("observation must have an observation ID")
        if payload.get("evidence_kind") not in EVIDENCE_KINDS:
            raise ProbeHarnessError("observation has an unsupported evidence kind")
        self._append("observation.completed", payload)

    def attach_observation_artifact(
        self,
        *,
        trial_id: str,
        observation_id: str,
        artifact: Mapping[str, Any],
    ) -> None:
        self._append(
            "observation.artifact_saved",
            {
                "trial_id": trial_id,
                "observation_id": observation_id,
                "activation_artifact": dict(artifact),
            },
        )

    def record_direction_bundle(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload.get("digest"), str):
            raise ProbeHarnessError("direction bundle has no content identity")
        if not isinstance(payload.get("run_directory"), str):
            raise ProbeHarnessError("direction bundle has no run directory")
        self._append("direction_bundle.loaded", payload)

    def state(self, *, records: list[dict[str, Any]] | None = None) -> SessionState:
        values = self.events.read_records() if records is None else records
        if not values:
            raise ProbeHarnessError("probe session contains no events")
        seen_events: set[str] = set()
        started_trials: set[str] = set()
        trials: dict[str, TrialView] = {}
        failed_trials: dict[str, dict[str, Any]] = {}
        observations: dict[str, dict[str, Any]] = {}
        observation_order: list[str] = []
        direction_bundles: list[dict[str, Any]] = []
        subject: dict[str, Any] | None = None
        settings: dict[str, Any] | None = None

        for index, event in enumerate(values):
            validate_event(event, session_id=self.session_id)
            event_id = str(event["event_id"])
            if event_id in seen_events:
                raise ProbeHarnessError(f"duplicate session event ID at record {index}")
            seen_events.add(event_id)
            event_type = str(event["event_type"])
            payload = dict(event["payload"])

            if event_type == "session.started":
                if index != 0 or subject is not None:
                    raise ProbeHarnessError("session.started must occur exactly once at record zero")
                if not isinstance(payload.get("subject"), Mapping):
                    raise ProbeHarnessError("session.started has no subject identity")
                if not isinstance(payload.get("settings"), Mapping):
                    raise ProbeHarnessError("session.started has no settings")
                subject = dict(payload["subject"])
                settings = dict(payload["settings"])
                continue

            if event_type == "direction_bundle.loaded":
                if not isinstance(payload.get("digest"), str) or not isinstance(
                    payload.get("run_directory"), str
                ):
                    raise ProbeHarnessError("direction_bundle.loaded has no valid identity")
                direction_bundles.append(payload)
                continue

            trial_id = payload.get("trial_id")
            if not isinstance(trial_id, str) or not trial_id:
                raise ProbeHarnessError(f"{event_type} has no trial ID")
            if event_type == "trial.started":
                if trial_id in started_trials:
                    raise ProbeHarnessError(f"trial started more than once: {trial_id}")
                for field_name in ("parent_trial_id", "replay_of"):
                    reference = payload.get(field_name)
                    if reference is not None and reference not in trials:
                        raise ProbeHarnessError(
                            f"trial {field_name} does not reference an earlier completed trial: "
                            f"{reference}"
                        )
                started_trials.add(trial_id)
            elif event_type == "trial.completed":
                if trial_id not in started_trials or trial_id in trials:
                    raise ProbeHarnessError(f"trial completion is out of order: {trial_id}")
                trials[trial_id] = TrialView(trial_id=trial_id, payload=payload)
            elif event_type == "trial.failed":
                if trial_id not in started_trials:
                    raise ProbeHarnessError(f"trial failure has no start event: {trial_id}")
                failed_trials[trial_id] = payload
            elif event_type == "trial.annotated":
                if trial_id not in trials:
                    raise ProbeHarnessError(f"annotation references an incomplete trial: {trial_id}")
                trial = trials[trial_id]
                if "note" in payload:
                    trial.notes.append(str(payload["note"]))
                if "tag" in payload:
                    trial.tags.add(str(payload["tag"]))
                if "marked" in payload:
                    trial.marked = bool(payload["marked"])
            elif event_type == "observation.completed":
                if trial_id not in trials:
                    raise ProbeHarnessError(f"observation references an incomplete trial: {trial_id}")
                observation_id = payload.get("observation_id")
                if not isinstance(observation_id, str) or not observation_id:
                    raise ProbeHarnessError("observation has no observation ID")
                if observation_id in observations:
                    raise ProbeHarnessError(f"duplicate observation ID: {observation_id}")
                if payload.get("evidence_kind") not in EVIDENCE_KINDS:
                    raise ProbeHarnessError(
                        f"observation has unsupported evidence kind: {payload.get('evidence_kind')!r}"
                    )
                trials[trial_id].observations.append(payload)
                observations[observation_id] = payload
                observation_order.append(observation_id)
            elif event_type == "observation.artifact_saved":
                observation_id = payload.get("observation_id")
                observation = observations.get(str(observation_id))
                if observation is None or observation.get("trial_id") != trial_id:
                    raise ProbeHarnessError(
                        "observation artifact does not reference an earlier observation"
                    )
                if observation.get("activation_artifact") is not None:
                    raise ProbeHarnessError(
                        f"observation already has an activation artifact: {observation_id}"
                    )
                artifact = payload.get("activation_artifact")
                if not isinstance(artifact, Mapping):
                    raise ProbeHarnessError("observation artifact reference is not an object")
                observation["activation_artifact"] = dict(artifact)

        if subject is None or settings is None:
            raise ProbeHarnessError("session has no valid start event")
        return SessionState(
            session_id=self.session_id,
            subject=subject,
            settings=settings,
            trials=trials,
            failed_trials=failed_trials,
            observations=observations,
            observation_order=observation_order,
            direction_bundles=direction_bundles,
        )

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = make_event(session_id=self.session_id, event_type=event_type, payload=payload)
        validate_event(event, session_id=self.session_id)
        self.events.append(
            [event],
            metadata={
                "domain": "probe-harness",
                "event_type": event_type,
            },
        )
