"""Console controller for a resident HF subject and durable probe session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contracts import new_id
from .errors import ProbeHarnessError
from .session_store import SessionState, SessionStore, TrialView

if TYPE_CHECKING:
    from .direction_bundle import DirectionBundle
    from .hf_subject import HFSubject


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = Path(
    "private/experiments/surgery/local-llama-guard-3-1b-mirror.yaml"
)
DEFAULT_CACHE_ROOT = Path(".scratch/cache/huggingface")
DEFAULT_SESSION_ROOT = Path("outputs/probe_sessions")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _layer_list(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("layers must contain non-negative integers")
    return layers


def _position_mode(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in {"all", "decision"}:
        raise argparse.ArgumentTypeError("positions must be either 'decision' or 'all'")
    return normalized


def _repository_path(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY / candidate
    return candidate.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-cli hf",
        description=(
            "Keep one pinned Hugging Face checkpoint resident while recording durable "
            "independent probe trials and exact-token observations."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=str(DEFAULT_EXPERIMENT),
        help="repository-relative pinned surgery experiment used as the subject specification",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="repository-local Hugging Face home containing the hub cache",
    )
    parser.add_argument(
        "--session-root",
        default=str(DEFAULT_SESSION_ROOT),
        help="repository-relative root for new probe sessions",
    )
    parser.add_argument(
        "--resume",
        help="existing session directory or session ID under --session-root",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the pinned snapshot to be downloaded into --cache-root",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int)
    parser.add_argument(
        "--prompt",
        action="append",
        help="submit a prompt and exit instead of opening the console; repeatable",
    )
    parser.add_argument(
        "--inspect-layers",
        type=_layer_list,
        help="inspect these comma-separated layers after each --prompt",
    )
    parser.add_argument(
        "--inspect-positions",
        type=_position_mode,
        default="decision",
        help="capture only the decision position or all exact input positions",
    )
    parser.add_argument(
        "--directions",
        help="load and validate a complete guard-study run directory at startup",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="persist full captures made by --inspect-layers",
    )
    parser.add_argument("--show-json", action="store_true")
    return parser


class ProbeController:
    """Coordinate one subject with one append-only session."""

    def __init__(self, subject: HFSubject, session: SessionStore) -> None:
        from .artifacts import ArtifactStore

        self.subject = subject
        self.session = session
        self.artifacts = ArtifactStore(session.directory)
        self.current_trial_id: str | None = None
        state = session.state()
        latest = state.latest_trial
        if latest is not None:
            self.current_trial_id = latest.trial_id
        self.current_observation_id: str | None = None
        try:
            latest_observation = state.resolve_observation(
                None,
                evidence_kind="observational",
            )
        except ProbeHarnessError:
            latest_observation = None
        if latest_observation is not None:
            self.current_observation_id = str(latest_observation["observation_id"])
        self.branch_parent_id: str | None = None
        self.direction_bundle: DirectionBundle | None = None
        self._pending_captures: dict[str, dict[str, Any]] = {}

    def submit(
        self,
        prompt: str,
        *,
        parent_trial_id: str | None = None,
        replay_of: str | None = None,
        replay_trial: TrialView | None = None,
    ) -> TrialView:
        generation_settings = {
            "max_new_tokens": self.subject.max_new_tokens,
            "do_sample": False,
        }
        trial_id = self.session.start_trial(
            prompt=prompt,
            parent_trial_id=parent_trial_id,
            replay_of=replay_of,
            generation=generation_settings,
        )
        try:
            if replay_trial is None:
                generated = self.subject.generate(prompt)
            else:
                source = replay_trial.payload
                generated = self.subject.generate(
                    prompt,
                    stored_inputs=source["prompt_inputs"],
                    stored_input_dtypes=source["prompt_input_dtypes"],
                    stored_rendered_prompt=str(source["rendered_prompt"]),
                )
            payload = {
                "trial_id": trial_id,
                "parent_trial_id": parent_trial_id,
                "replay_of": replay_of,
                "subject_digest": self.subject.identity["digest"],
                **generated,
            }
            self.session.complete_trial(trial_id, payload)
        except BaseException as error:
            self.session.fail_trial(trial_id, error)
            raise
        self.current_trial_id = trial_id
        self.branch_parent_id = None
        return self.session.state().resolve_trial(trial_id)

    def replay(self, value: str | None = None) -> TrialView:
        source = self.session.state().resolve_trial(value)
        return self.submit(
            str(source.payload["prompt"]),
            parent_trial_id=source.trial_id,
            replay_of=source.trial_id,
            replay_trial=source,
        )

    def inspect(
        self,
        value: str | None = None,
        *,
        layers: Sequence[int] | None = None,
        positions: str = "decision",
    ) -> dict[str, Any]:
        trial = self.session.state().resolve_trial(value)
        selected = tuple(layers) if layers is not None else (len(self.subject.layers) - 1,)
        observation, tensors = self.subject.inspect(
            trial.payload,
            capture_layers=selected,
            positions=positions,
        )
        observation_id = new_id("observation")
        payload = {
            "observation_id": observation_id,
            "trial_id": trial.trial_id,
            "subject_digest": self.subject.identity["digest"],
            **observation,
            "activation_artifact": None,
        }
        self.session.record_observation(payload)
        self._pending_captures = {observation_id: dict(tensors)}
        self.current_trial_id = trial.trial_id
        self.current_observation_id = observation_id
        return payload

    def load_directions(self, run_directory: str | Path) -> dict[str, Any]:
        from .direction_bundle import DirectionBundle

        bundle = DirectionBundle.load(
            run_directory,
            subject_identity=self.subject.identity,
        )
        state = self.session.state()
        if not state.direction_bundles or (
            state.direction_bundles[-1].get("digest") != bundle.identity["digest"]
        ):
            self.session.record_direction_bundle(bundle.identity)
        self.direction_bundle = bundle
        return bundle.summary()

    def _observational_payload(self, value: str | None = None) -> dict[str, Any]:
        target = value if value is not None else self.current_observation_id
        observation = self.session.state().resolve_observation(
            target,
            evidence_kind="observational",
        )
        self.current_observation_id = str(observation["observation_id"])
        self.current_trial_id = str(observation["trial_id"])
        return observation

    def _activation_tensors(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        observation_id = str(observation["observation_id"])
        pending = self._pending_captures.get(observation_id)
        if pending is not None:
            return pending
        artifact = observation.get("activation_artifact")
        if isinstance(artifact, Mapping):
            return self.artifacts.load_activations(artifact)
        raise ProbeHarnessError(
            "full tensors for this observation are no longer resident; rerun :inspect "
            "or save them before restarting"
        )

    def save_activations(self, value: str | None = None) -> dict[str, Any]:
        observation = self._observational_payload(value)
        existing = observation.get("activation_artifact")
        if isinstance(existing, Mapping):
            return dict(existing)
        tensors = self._activation_tensors(observation)
        artifact = self.artifacts.save_activations(tensors)
        if artifact is None:
            raise ProbeHarnessError("observation has no activation tensors to save")
        self.session.attach_observation_artifact(
            trial_id=str(observation["trial_id"]),
            observation_id=str(observation["observation_id"]),
            artifact=artifact,
        )
        return artifact

    def project(
        self,
        direction_name: str,
        *,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        if self.direction_bundle is None:
            raise ProbeHarnessError("load a direction bundle with :load-directions first")
        observation = self._observational_payload(observation_id)
        entry = self.direction_bundle.resolve(direction_name)
        if entry.layer_index is None:
            raise ProbeHarnessError(
                f"direction {entry.name} does not target a transformer-layer capture"
            )
        tensor_key = f"layer.{entry.layer_index}.{observation['positions']}"
        tensors = self._activation_tensors(observation)
        activation = tensors.get(tensor_key)
        if activation is None:
            raise ProbeHarnessError(
                f"observation did not capture layer {entry.layer_index}; rerun :inspect "
                f"with --layers {entry.layer_index}"
            )
        projection = self.direction_bundle.project(entry.name, activation)
        token_positions = observation.get("token_positions")
        profile = projection.get("profile")
        if not isinstance(token_positions, list) or not isinstance(profile, list):
            raise ProbeHarnessError("observation has no aligned token-position profile")
        if len(token_positions) != len(profile):
            raise ProbeHarnessError("direction projection does not align with captured token positions")
        projection["profile"] = [
            {**projected, **token}
            for projected, token in zip(profile, token_positions, strict=True)
        ]
        projection_observation_id = new_id("observation")
        payload = {
            "observation_id": projection_observation_id,
            "trial_id": str(observation["trial_id"]),
            "parent_observation_id": str(observation["observation_id"]),
            "subject_digest": self.subject.identity["digest"],
            "evidence_kind": "geometric-estimate",
            "operation": "direction-projection",
            "direction_bundle": {
                "digest": self.direction_bundle.identity["digest"],
                "run_directory": self.direction_bundle.identity["run_directory"],
                "dataset_sha256": self.direction_bundle.identity["dataset_sha256"],
                "metric": self.direction_bundle.identity["metric"],
            },
            "projection": projection,
        }
        self.session.record_observation(payload)
        return payload

    def logits(self, value: str | None = None) -> dict[str, Any]:
        observation = self._observational_payload(value)
        logits = observation.get("logits")
        if not isinstance(logits, Mapping):
            raise ProbeHarnessError("observation has no compact logit summary")
        return dict(logits)

    def annotate_current(
        self,
        *,
        note: str | None = None,
        tag: str | None = None,
        marked: bool | None = None,
    ) -> TrialView:
        trial = self.session.state().resolve_trial(self.current_trial_id)
        self.session.annotate(trial.trial_id, note=note, tag=tag, marked=marked)
        return self.session.state().resolve_trial(trial.trial_id)


def _emit_trial(trial: TrialView, *, show_json: bool) -> None:
    completion = str(trial.payload.get("completion_text", "")).strip()
    print(completion if completion else "<empty completion>")
    print(f"[{trial.trial_id}]", file=sys.stderr)
    if show_json:
        print(json.dumps(trial.payload, indent=2, ensure_ascii=False, sort_keys=True))


def _emit_observation(observation: dict[str, Any], *, show_json: bool) -> None:
    print(
        "inspect> "
        f"label={observation['predicted_label'] or '-'} "
        f"margin={observation['unsafe_minus_safe_logit_margin']:+.6f} "
        f"top={observation['top_token_text']!r} "
        f"positions={observation['positions']}"
    )
    summaries = observation["activation_summaries"]
    for layer in observation["capture_layers"]:
        summary = summaries[str(layer)]
        print(
            f"  layer {layer}: shape={summary['shape']} "
            f"norm={summary['l2_norm']:.6f} mean={summary['mean']:+.6f} "
            f"std={summary['std']:.6f}"
        )
    artifact = observation.get("activation_artifact")
    if artifact is not None:
        print(f"  artifact: {artifact['path']}")
    if show_json:
        print(json.dumps(observation, indent=2, ensure_ascii=False, sort_keys=True))


def _emit_direction_bundle(bundle: Mapping[str, Any], *, show_json: bool) -> None:
    print(
        "directions> "
        f"digest={bundle['digest']} layers={bundle['num_layers']} "
        f"hidden={bundle['hidden_size']} metric={bundle['metric']}"
    )
    learned = [
        value["name"]
        for value in bundle["directions"]
        if value.get("kind") == "learned"
    ]
    print(f"  learned: {', '.join(learned)}")
    if show_json:
        print(json.dumps(bundle, indent=2, ensure_ascii=False, sort_keys=True))


def _emit_projection(observation: Mapping[str, Any], *, show_json: bool) -> None:
    projection = observation["projection"]
    direction = projection["direction"]
    print(
        "project> "
        f"{direction['name']} layer={direction['layer_index']} "
        f"last={projection['raw_projection_last']:+.6f} "
        f"range=[{projection['raw_projection_min']:+.6f}, "
        f"{projection['raw_projection_max']:+.6f}] "
        f"positions={projection['position_count']}"
    )
    last = projection["profile"][-1]
    print(
        f"  last token {last['position']} {last['token_text']!r}: "
        f"cos={last['cosine_similarity']:+.6f} "
        f"midpoint-z={last['midpoint_standardized_projection']:+.6f}"
    )
    if show_json:
        print(json.dumps(observation, indent=2, ensure_ascii=False, sort_keys=True))


def _emit_logits(logits: Mapping[str, Any], *, show_json: bool) -> None:
    safe = logits["safe"]
    unsafe = logits["unsafe"]
    print(
        "logits> "
        f"safe={safe['logit']:+.6f} unsafe={unsafe['logit']:+.6f} "
        f"margin={unsafe['logit'] - safe['logit']:+.6f}"
    )
    for token in logits["top_tokens"]:
        print(
            f"  {token['rank']:2d}. {token['token_id']:6d} "
            f"{token['token_text']!r} {token['logit']:+.6f}"
        )
    if show_json:
        print(json.dumps(logits, indent=2, ensure_ascii=False, sort_keys=True))


def _trial_summary(trial: TrialView) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "parent_trial_id": trial.payload.get("parent_trial_id"),
        "replay_of": trial.payload.get("replay_of"),
        "prompt": trial.payload.get("prompt"),
        "completion_text": trial.payload.get("completion_text"),
        "emitted_label": trial.payload.get("emitted_label"),
        "emitted_category": trial.payload.get("emitted_category"),
        "notes": list(trial.notes),
        "tags": sorted(trial.tags),
        "marked": trial.marked,
        "observation_ids": [value.get("observation_id") for value in trial.observations],
    }


def _print_history(state: SessionState) -> None:
    if not state.trials:
        print("No completed trials.")
        return
    for index, trial in enumerate(state.trials.values(), start=1):
        prompt = str(trial.payload.get("prompt", "")).replace("\n", " ")
        if len(prompt) > 72:
            prompt = prompt[:69] + "..."
        label = trial.payload.get("emitted_label") or "-"
        category = trial.payload.get("emitted_category") or "-"
        flags = "*" if trial.marked else " "
        print(f"{index:3d}{flags} {trial.trial_id}  {label}/{category}  {prompt}")


def _parse_inspect_command(
    command: str,
) -> tuple[str | None, tuple[int, ...] | None, str]:
    try:
        values = shlex.split(command)
    except ValueError as error:
        raise ProbeHarnessError(f"could not parse :inspect command: {error}") from error
    target: str | None = None
    layers: tuple[int, ...] | None = None
    positions = "decision"
    index = 1
    while index < len(values):
        value = values[index]
        if value == "--layers":
            index += 1
            if index >= len(values):
                raise ProbeHarnessError(":inspect --layers requires a value")
            try:
                layers = _layer_list(values[index])
            except argparse.ArgumentTypeError as error:
                raise ProbeHarnessError(str(error)) from error
        elif value.startswith("--layers="):
            try:
                layers = _layer_list(value.split("=", 1)[1])
            except argparse.ArgumentTypeError as error:
                raise ProbeHarnessError(str(error)) from error
        elif value == "--positions":
            index += 1
            if index >= len(values):
                raise ProbeHarnessError(":inspect --positions requires a value")
            try:
                positions = _position_mode(values[index])
            except argparse.ArgumentTypeError as error:
                raise ProbeHarnessError(str(error)) from error
        elif value.startswith("--positions="):
            try:
                positions = _position_mode(value.split("=", 1)[1])
            except argparse.ArgumentTypeError as error:
                raise ProbeHarnessError(str(error)) from error
        elif target is None:
            target = value
        else:
            raise ProbeHarnessError(f"unexpected :inspect argument: {value}")
        index += 1
    return target, layers, positions


def _parse_project_command(command: str) -> tuple[str, str | None]:
    try:
        values = shlex.split(command)
    except ValueError as error:
        raise ProbeHarnessError(f"could not parse :project command: {error}") from error
    if len(values) < 2:
        raise ProbeHarnessError(":project requires a direction name")
    direction = values[1]
    observation: str | None = None
    index = 2
    while index < len(values):
        value = values[index]
        if value == "--observation":
            index += 1
            if index >= len(values):
                raise ProbeHarnessError(":project --observation requires a value")
            observation = values[index]
        elif value.startswith("--observation="):
            observation = value.split("=", 1)[1]
        else:
            raise ProbeHarnessError(f"unexpected :project argument: {value}")
        index += 1
    return direction, observation


def _single_command_value(command: str, name: str, *, required: bool) -> str | None:
    value = command[len(name) :].strip()
    if not value:
        if required:
            raise ProbeHarnessError(f"{name} requires a value")
        return None
    try:
        values = shlex.split(value)
    except ValueError as error:
        raise ProbeHarnessError(f"could not parse {name} command: {error}") from error
    if len(values) != 1:
        raise ProbeHarnessError(f"{name} accepts exactly one value")
    return values[0]


def _print_help() -> None:
    print(":history                    list completed trials")
    print(":show [trial]               show one trial and its annotations")
    print(":replay [trial]             replay exact stored input tokens")
    print(":branch [trial]             make the next prompt a child of this trial")
    print(":note <text>                annotate the current trial")
    print(":tag <token>                tag the current trial")
    print(":mark                       mark the current trial as interesting")
    print(":load-directions <run-dir>  validate a complete guard-study direction bundle")
    print(":directions                  list the active bundle's named directions")
    print(":inspect [trial] [--layers 12,14] [--positions decision|all]")
    print(":project <direction> [--observation id]")
    print(":logits [observation]        show compact decision logits")
    print(":save-activations [observation]")
    print(":multiline                  read a prompt until a line containing :end")
    print(":session                    print the resumable session directory")
    print(":quit                       exit the console")
    print("Prefix a literal leading colon with another colon, for example ::topic.")


def _is_command(value: str, name: str) -> bool:
    return value == name or value.startswith(name + " ")


def _run_multiline() -> str:
    print("Enter prompt text. Finish with :end on its own line.")
    lines: list[str] = []
    while True:
        try:
            value = input("...> ")
        except EOFError:
            break
        if value.strip().casefold() == ":end":
            break
        lines.append(value)
    prompt = "\n".join(lines)
    if not prompt.strip():
        raise ProbeHarnessError("multiline prompt must not be empty")
    return prompt


def run_console(
    controller: ProbeController,
    *,
    show_json: bool,
) -> int:
    print("Ready. Llama Guard prompts are independent trials. Use :help or :quit.")
    print(f"Session: {controller.session.directory}")
    while True:
        try:
            prompt = input("probe> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt.strip():
            continue
        command = prompt.strip()
        folded = command.casefold()
        if folded in {":quit", ":exit"}:
            return 0
        try:
            if folded == ":help":
                _print_help()
            elif folded == ":history":
                _print_history(controller.session.state())
            elif _is_command(folded, ":show"):
                value = command[len(":show") :].strip() or None
                trial = controller.session.state().resolve_trial(value)
                print(json.dumps(_trial_summary(trial), indent=2, ensure_ascii=False))
                controller.current_trial_id = trial.trial_id
            elif _is_command(folded, ":replay"):
                value = command[len(":replay") :].strip() or None
                _emit_trial(controller.replay(value), show_json=show_json)
            elif _is_command(folded, ":branch"):
                value = command[len(":branch") :].strip() or None
                trial = controller.session.state().resolve_trial(value)
                controller.branch_parent_id = trial.trial_id
                controller.current_trial_id = trial.trial_id
                print(f"Next prompt will branch from {trial.trial_id}.")
            elif _is_command(folded, ":note"):
                value = command[len(":note") :]
                trial = controller.annotate_current(note=value)
                print(f"Annotated {trial.trial_id}.")
            elif _is_command(folded, ":tag"):
                value = command[len(":tag") :]
                trial = controller.annotate_current(tag=value)
                print(f"Tagged {trial.trial_id}.")
            elif folded == ":mark":
                trial = controller.annotate_current(marked=True)
                print(f"Marked {trial.trial_id}.")
            elif _is_command(folded, ":load-directions"):
                value = _single_command_value(command, ":load-directions", required=True)
                bundle = controller.load_directions(_repository_path(str(value)))
                _emit_direction_bundle(bundle, show_json=show_json)
            elif folded == ":directions":
                if controller.direction_bundle is None:
                    raise ProbeHarnessError("no direction bundle is loaded")
                _emit_direction_bundle(controller.direction_bundle.summary(), show_json=show_json)
            elif _is_command(folded, ":inspect"):
                target, layers, positions = _parse_inspect_command(command)
                observation = controller.inspect(target, layers=layers, positions=positions)
                _emit_observation(observation, show_json=show_json)
            elif _is_command(folded, ":project"):
                direction, observation_id = _parse_project_command(command)
                observation = controller.project(
                    direction,
                    observation_id=observation_id,
                )
                _emit_projection(observation, show_json=show_json)
            elif _is_command(folded, ":logits"):
                value = _single_command_value(command, ":logits", required=False)
                _emit_logits(controller.logits(value), show_json=show_json)
            elif _is_command(folded, ":save-activations"):
                value = _single_command_value(command, ":save-activations", required=False)
                artifact = controller.save_activations(value)
                print(f"activations> {artifact['path']} ({artifact['sha256']})")
            elif folded == ":multiline":
                value = _run_multiline()
                trial = controller.submit(value, parent_trial_id=controller.branch_parent_id)
                _emit_trial(trial, show_json=show_json)
            elif folded in {":session", ":resume"}:
                print(controller.session.directory)
            else:
                if prompt.startswith("::"):
                    prompt = prompt[1:]
                elif prompt.startswith(":"):
                    raise ProbeHarnessError(f"unknown command: {command.split()[0]}")
                trial = controller.submit(prompt, parent_trial_id=controller.branch_parent_id)
                _emit_trial(trial, show_json=show_json)
        except KeyboardInterrupt:
            print("\nInterrupted; installed hooks were removed and the failed trial was recorded.")
        except ProbeHarnessError as error:
            print(f"probe-harness: {error}", file=sys.stderr)


def _resolve_resume(value: str, session_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    repository_candidate = REPOSITORY / candidate
    if repository_candidate.exists():
        return repository_candidate.resolve()
    root_candidate = session_root / candidate
    if root_candidate.exists():
        return root_candidate.resolve()
    raise ProbeHarnessError(f"probe session does not exist: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_root = _repository_path(args.session_root)
    print("Resolving and loading pinned HF subject ...", file=sys.stderr)
    subject: HFSubject | None = None
    try:
        from .hf_subject import HFSubject

        subject = HFSubject.load_from_experiment(
            args.experiment,
            repository=REPOSITORY,
            cache_root=args.cache_root,
            allow_download=args.allow_download,
            max_new_tokens=args.max_new_tokens,
        )
        if args.resume:
            session = SessionStore.resume(
                _resolve_resume(args.resume, session_root),
                expected_subject_digest=str(subject.identity["digest"]),
            )
            print(f"Resumed session {session.session_id}.", file=sys.stderr)
        else:
            session = SessionStore.create(
                session_root,
                subject=subject.identity,
                settings={"history_mode": "independent", "schema_version": 1},
            )
            print(f"Created session {session.session_id}.", file=sys.stderr)
        controller = ProbeController(subject, session)
        if args.directions:
            bundle = controller.load_directions(_repository_path(args.directions))
            _emit_direction_bundle(bundle, show_json=args.show_json)
        if args.prompt:
            for index, prompt in enumerate(args.prompt):
                if index:
                    print()
                trial = controller.submit(prompt)
                _emit_trial(trial, show_json=args.show_json)
                if args.inspect_layers is not None:
                    observation = controller.inspect(
                        trial.trial_id,
                        layers=args.inspect_layers,
                        positions=args.inspect_positions,
                    )
                    _emit_observation(observation, show_json=args.show_json)
                    if args.save_activations:
                        artifact = controller.save_activations(
                            str(observation["observation_id"])
                        )
                        print(
                            f"activations> {artifact['path']} ({artifact['sha256']})"
                        )
            print(f"Session: {session.directory}", file=sys.stderr)
            return 0
        return run_console(controller, show_json=args.show_json)
    except ProbeHarnessError as error:
        print(f"probe-harness: {error}", file=sys.stderr)
        return 2
    finally:
        if subject is not None:
            subject.close()
