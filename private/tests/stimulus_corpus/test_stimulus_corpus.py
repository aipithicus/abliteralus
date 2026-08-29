"""Contract tests for the stimulus corpus package."""

from __future__ import annotations

import pytest

from stimulus_corpus import (
    ArxivTaxonomySource,
    HeadDecomposition,
    ModelCardSource,
    StimulusFormatError,
    StimulusStore,
    group_values,
    TermRules,
    atom_expression,
    build_attestation,
    compose_expression,
    derive_stimulus_id,
    expression_surface,
    normalize_surface,
)
from stimulus_corpus.errors import StimulusSourceError
from stimulus_corpus.sources.base import Candidate

CARD = """# Llama Guard 3 Model Card

## Hazard Taxonomy

### S1: Violent Crimes
AI models should not create content that enables or encourages violent crimes.
Examples include: terrorism; genocide; murder; assault; battery; kidnapping.

### S9: Indiscriminate Weapons
Responses should not enable the creation of indiscriminate weapons.
Chemical weapons; biological weapons; nerve gas; anthrax; cobalt bombs;
fission bombs; cluster munitions.

### S13: Elections
Factually incorrect election information; time of voting; place of voting.

## Evaluation

<table><tr><td>english</td><td>0.939/0.040</td></tr>
<tr><td>vietnamese</td><td>0.723/0.130</td></tr></table>
"""


#: The shape Llama Guard 3 actually uses: bold ``**S1 - Title**`` labels, one
#: long policy sentence per section, terms carried in enumerated clauses and
#: ``(ex: ...)`` groups rather than on separate lines.
STRUCTURED_CARD = """## Hazard Taxonomy and Policy

**S1 - Violent Crimes**

Responses that enable, encourage, or endorse the commission of violent crimes, \
including: (1) unlawful violence toward people (ex: terrorism, genocide, murder) \
and (2) unlawful violence toward animals (ex: animal abuse)

**S9 - Indiscriminate Weapons**

Responses that enable, encourage, or endorse the creation of indiscriminate \
weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological \
weapons (ex: anthrax)

## Evaluation

<table><tr><td>english</td><td>0.939/0.040</td></tr></table>
"""


@pytest.fixture()
def structured_card(tmp_path):
    path = tmp_path / "structured-card.md"
    path.write_text(STRUCTURED_CARD, encoding="utf-8")
    return ModelCardSource(path, source_id="card.structured")


@pytest.fixture()
def card(tmp_path):
    path = tmp_path / "model-card.md"
    path.write_text(CARD, encoding="utf-8")
    return path


@pytest.fixture()
def source(card):
    return ModelCardSource(card, source_id="card.test-guard")


# --- identity -------------------------------------------------------------


def test_normalization_folds_case_and_whitespace():
    assert normalize_surface("  Nerve   Gas\n") == "nerve gas"


def test_normalization_rejects_empty_surface():
    with pytest.raises(StimulusFormatError):
        normalize_surface("   ")


def test_identity_is_a_function_of_the_expression_only():
    left = derive_stimulus_id(atom_expression("Virology"), corpus_version="v1")
    right = derive_stimulus_id(atom_expression("  virology "), corpus_version="v1")
    assert left == right, "normalized surfaces must agree on identity"


def test_identity_separates_distinct_surfaces_and_corpus_versions():
    virology = derive_stimulus_id(atom_expression("virology"), corpus_version="v1")
    topology = derive_stimulus_id(atom_expression("topology"), corpus_version="v1")
    bumped = derive_stimulus_id(atom_expression("virology"), corpus_version="v2")
    assert len({virology, topology, bumped}) == 3


def test_attestation_rejects_unknown_projection_source():
    with pytest.raises(StimulusFormatError):
        build_attestation(
            surface="virology",
            corpus_version="v1",
            source_id="card.test",
            source_version="sha256:abc",
            source_ref="s9@0-8",
            projection_source="vibes",
            run_id="run",
        )


def test_attestation_carries_no_label_fields():
    record = build_attestation(
        surface="virology",
        corpus_version="v1",
        source_id="card.test",
        source_version="sha256:abc",
        source_ref="s9@0-8",
        projection_source="literal",
        run_id="run",
    )
    forbidden = {"expected", "label", "hazard", "a_priori_hazard_association"}
    assert forbidden.isdisjoint(record)


# --- extraction -----------------------------------------------------------


def test_sections_are_located_by_code(source):
    assert [section.code for section in source.sections()] == ["s1", "s9", "s13"]


def test_literal_terms_survive_extraction(source):
    surfaces = {candidate.surface for candidate in source.extract().candidates}
    for expected in ("nerve gas", "anthrax", "cobalt bombs", "cluster munitions"):
        assert expected in surfaces


def test_policy_prose_is_rejected_with_a_reason(source):
    result = source.extract()
    surfaces = {candidate.surface for candidate in result.candidates}
    assert not any("should not" in surface for surface in surfaces)
    assert not any(surface.startswith("examples") for surface in surfaces)
    assert result.reason_counts().get("policy-language", 0) > 0


def test_rejections_are_reported_not_discarded(source):
    result = source.extract()
    assert result.rejected, "a silent filter is indistinguishable from an empty source"
    assert all(rejection.reason for rejection in result.rejected)


def test_candidates_carry_resolvable_provenance(source, card):
    text = card.read_text(encoding="utf-8")
    sections = {section.code: section for section in source.sections()}
    for candidate in source.extract().candidates:
        code, _, span = candidate.source_ref.partition("@")
        start = int(span.split("-")[0])
        section = sections[code]
        assert section.start <= start < section.end
        assert candidate.surface in text[section.start : section.end].casefold()


def test_last_section_stops_at_the_next_heading(source):
    """Without a terminator the final section absorbs the rest of the file."""

    sections = {section.code: section for section in source.sections()}
    surfaces = {candidate.surface for candidate in source.extract().candidates}
    assert "time of voting" in surfaces, "the real S13 terms must survive"
    assert not {"english", "vietnamese"} & surfaces, "evaluation table leaked in"
    assert "## Evaluation" not in source._text[sections["s13"].start : sections["s13"].end]


def test_html_fragments_are_rejected(tmp_path):
    card = tmp_path / "html.md"
    card.write_text(
        "**S1 - Violent Crimes**\n\nResponses including: (ex: arson)\n<td>english</td>\n",
        encoding="utf-8",
    )
    result = ModelCardSource(card, source_id="card.html").extract()
    surfaces = {candidate.surface for candidate in result.candidates}
    assert "arson" in surfaces
    assert not any("<" in surface for surface in surfaces)


def test_bold_section_labels_are_recognised(structured_card):
    assert [section.code for section in structured_card.sections()] == ["s1", "s9"]
    assert structured_card.sections()[0].title == "Violent Crimes"


def test_clauses_and_examples_are_extracted_at_both_granularities(structured_card):
    roles: dict[str, list[str]] = {}
    for candidate in structured_card.extract().candidates:
        roles.setdefault(candidate.extra["role"], []).append(candidate.surface)
    # The clause names the class, its example group names the members.
    assert "chemical weapons" in roles["clause"]
    assert "nerve gas" in roles["example"]
    assert "unlawful violence toward people" in roles["clause"]
    assert {"terrorism", "genocide", "murder"} <= set(roles["example"])


def test_policy_sentence_does_not_survive_the_residue_pass(structured_card):
    result = structured_card.extract()
    assert "prose" not in {c.extra["role"] for c in result.candidates}
    assert result.reason_counts().get("too-many-words", 0) > 0


def test_section_pattern_failure_is_explicit(tmp_path):
    empty = tmp_path / "nothing.md"
    empty.write_text("# A card with no hazard sections\n", encoding="utf-8")
    with pytest.raises(StimulusSourceError, match="no hazard-category sections"):
        ModelCardSource(empty, source_id="card.empty").sections()


def test_comma_splitting_is_opt_in(card):
    default = ModelCardSource(card, source_id="card.test-guard").extract()
    commas = ModelCardSource(
        card, source_id="card.test-guard", rules=TermRules.with_comma_splitting()
    ).extract()
    assert len(commas.candidates) >= len(default.candidates)


# --- store ----------------------------------------------------------------


def test_append_and_fold_round_trip(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    receipt, stats = store.append_from_source(source, corpus_version="v1")
    assert receipt.appended_records == stats["kept"]
    stimuli = store.stimuli()
    assert len(stimuli) == receipt.appended_records
    assert all(item.review_status == "proposed" for item in stimuli)


def test_two_sources_attesting_one_surface_fold_to_one_stimulus(tmp_path):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    shared = [Candidate(surface="virology", source_ref="s9@1-9", projection_source="literal")]
    store.append_candidates(
        shared,
        source_id="card.test-guard",
        source_version="sha256:aaa",
        corpus_version="v1",
    )
    store.append_candidates(
        [Candidate(surface="virology", source_ref="q-bio", projection_source="ontological")],
        source_id="taxonomy.arxiv",
        source_version="2026-08-29",
        corpus_version="v1",
    )
    stimuli = store.stimuli()
    assert len(stimuli) == 1
    only = stimuli[0]
    assert only.sources == ("card.test-guard", "taxonomy.arxiv")
    assert only.projection_sources == ("literal", "ontological")
    assert len(only.attestations) == 2


def test_partition_read_narrows_to_one_source(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_from_source(source, corpus_version="v1")
    store.append_candidates(
        [Candidate(surface="topology", source_ref="math.GN", projection_source="ontological")],
        source_id="taxonomy.arxiv",
        source_version="2026-08-29",
        corpus_version="v1",
    )
    narrowed = store.stimuli(source_id="taxonomy.arxiv")
    assert [item.surface for item in narrowed] == ["topology"]


def test_empty_extraction_is_refused(tmp_path):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    with pytest.raises(StimulusFormatError, match="empty extraction pass"):
        store.append_candidates(
            [],
            source_id="card.test",
            source_version="sha256:abc",
            corpus_version="v1",
        )


def test_export_is_deterministic_over_the_same_boundary(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_from_source(source, corpus_version="v1", review_status="accepted")
    first = store.export_dataset(
        tmp_path / "a.yaml", name="battery", description="", corpus_version="v1"
    )
    second = store.export_dataset(
        tmp_path / "b.yaml", name="battery", description="", corpus_version="v1"
    )
    assert first["content_sha256"] == second["content_sha256"]
    assert first["stimulus_count"] > 0


def test_export_refuses_when_nothing_is_accepted(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_from_source(source, corpus_version="v1")
    with pytest.raises(StimulusFormatError, match="nothing to export"):
        store.export_dataset(
            tmp_path / "out.yaml", name="battery", description="", corpus_version="v1"
        )


# --- composition ----------------------------------------------------------


def test_compose_expression_is_distinct_from_the_atom_of_the_same_surface():
    """Constructed and attested are different objects even when they read alike."""

    composed = compose_expression([("modifier", "chemical"), ("head", "weapons")])
    assert expression_surface(composed) == "chemical weapons"
    assert derive_stimulus_id(composed, corpus_version="v1") != derive_stimulus_id(
        atom_expression("chemical weapons"), corpus_version="v1"
    )


def test_head_decomposition_discovers_a_corroborated_basis():
    source = HeadDecomposition(
        [
            "chemical weapons",
            "biological weapons",
            "nuclear weapons",
            "property crimes",
            "cyber crimes",
        ],
        source_id="decompose.test",
        source_version="boundary:1",
    )
    basis = source.basis()
    assert set(basis) == {"weapons", "crimes"}
    assert set(basis["weapons"]) == {"chemical", "biological", "nuclear"}


def test_head_decomposition_requires_corroboration():
    """One term sharing a head is a coincidence, not a vocabulary."""

    source = HeadDecomposition(
        ["cobalt bombs", "chemical weapons", "biological weapons"],
        source_id="decompose.test",
        source_version="boundary:1",
    )
    assert set(source.basis()) == {"weapons"}
    reasons = source.extract().reason_counts()
    assert reasons.get("head-not-corroborated") == 1


def test_head_decomposition_emits_atoms_not_the_cross_product():
    source = HeadDecomposition(
        ["chemical weapons", "biological weapons", "cyber crimes", "drug crimes"],
        source_id="decompose.test",
        source_version="boundary:1",
    )
    surfaces = {candidate.surface for candidate in source.extract().candidates}
    assert {"weapons", "crimes", "chemical", "biological", "cyber", "drug"} == surfaces
    assert "chemical crimes" not in surfaces, "cross-head composition needs review"


def test_head_decomposition_skips_surfaces_already_in_the_corpus():
    source = HeadDecomposition(
        ["chemical weapons", "biological weapons", "weapons"],
        source_id="decompose.test",
        source_version="boundary:1",
    )
    surfaces = {candidate.surface for candidate in source.extract().candidates}
    assert "weapons" not in surfaces, "already attested; decomposition must not duplicate it"
    assert {"chemical", "biological"} <= surfaces


def test_composed_stimulus_round_trips_through_the_store(tmp_path):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_candidates(
        [
            Candidate(
                surface="chemical weapons",
                source_ref="generated",
                projection_source="lexical",
                expression=compose_expression([("modifier", "chemical"), ("head", "weapons")]),
            )
        ],
        source_id="gen.test",
        source_version="v1",
        corpus_version="v1",
    )
    only = store.stimuli()[0]
    assert only.surface == "chemical weapons"
    assert only.expression["kind"] == "compose"
    assert [part["role"] for part in only.expression["parts"]] == ["modifier", "head"]


# --- groups versus split key ----------------------------------------------


def test_groups_are_free_and_plural_but_names_are_identifiers():
    record = build_attestation(
        surface="anthrax",
        corpus_version="v1",
        source_id="card.test",
        source_version="sha256:abc",
        source_ref="s9@0-7",
        projection_source="literal",
        run_id="run",
        groups={"policy_category": "s9", "composition_role": "Head Term!"},
    )
    assert record["groups"] == {"policy_category": "s9", "composition_role": "Head Term!"}
    with pytest.raises(StimulusFormatError, match="group name"):
        build_attestation(
            surface="anthrax",
            corpus_version="v1",
            source_id="card.test",
            source_version="sha256:abc",
            source_ref="s9@0-7",
            projection_source="literal",
            run_id="run",
            groups={"Policy Category": "s9"},
        )


def test_split_key_stays_a_validated_identifier():
    """Groups are cosmetic; the split key decides what held-out means."""

    with pytest.raises(StimulusFormatError, match="split_key"):
        build_attestation(
            surface="anthrax",
            corpus_version="v1",
            source_id="card.test",
            source_version="sha256:abc",
            source_ref="s9@0-7",
            projection_source="literal",
            run_id="run",
            split_key="Life Sciences",
        )


def test_groups_merge_across_sources_and_split_key_stays_unset(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_from_source(source, corpus_version="v1")
    store.append_candidates(
        [
            Candidate(
                surface="anthrax",
                source_ref="decompose",
                projection_source="lexical",
                groups={"composition_role": "modifier"},
            )
        ],
        source_id="decompose.test",
        source_version="boundary:1",
        corpus_version="v1",
    )
    anthrax = next(item for item in store.stimuli() if item.surface == "anthrax")
    assert anthrax.groups == {"policy_category": "s9", "composition_role": "modifier"}
    assert anthrax.split_key is None, "grouping must never imply a split assignment"


def test_group_values_yields_a_stable_palette_mapping(tmp_path, source):
    store = StimulusStore(tmp_path / "corpus.jsonl")
    store.append_from_source(source, corpus_version="v1")
    palette = group_values(store.stimuli(), "policy_category")
    assert set(palette.values()) <= {"s1", "s9", "s13"}
    assert all(key.startswith("stim.v1.") for key in palette)


# --- distributive prepositional phrases -----------------------------------


PP_CARD = """**S13 - Elections**

Responses that contain factually incorrect information about electoral \
systems and processes, including in the time, place, or manner of voting in \
civic elections

## Evaluation
"""


def test_shared_prepositional_phrase_distributes_across_the_coordination(tmp_path):
    """The PP is elided from every member but the last, not absent."""

    card = tmp_path / "pp.md"
    card.write_text(PP_CARD, encoding="utf-8")
    result = ModelCardSource(card, source_id="card.pp").extract()
    surfaces = {candidate.surface for candidate in result.candidates}
    assert {"time of voting", "place of voting", "manner of voting"} <= surfaces
    assert "in the time" not in surfaces
    distributed = {
        candidate.surface
        for candidate in result.candidates
        if candidate.extra.get("distributed_tail") == "of voting"
    }
    assert len(distributed) == 3


def test_stranded_conjunction_does_not_survive_the_comma_split():
    """', or manner' must not yield a member beginning with 'or'."""

    from stimulus_corpus.sources.model_card import _EXAMPLE_ITEMS

    assert _EXAMPLE_ITEMS.split("a, b, or c") == ["a", "b", "c"]
    assert _EXAMPLE_ITEMS.split("a, b, and c") == ["a", "b", "c"]
    # the whitespace requirement after the conjunction protects real words
    assert _EXAMPLE_ITEMS.split("fraud, organized crime") == ["fraud", "organized crime"]
    assert _EXAMPLE_ITEMS.split("theft, andirons") == ["theft", "andirons"]


def test_distribution_is_refused_when_the_shape_is_ambiguous():
    from stimulus_corpus.sources.model_card import _distribute_tail

    # earlier members already carry prepositions: not an elided coordination
    pieces = [("denial of service attacks", 0), ("container escapes of note", 30)]
    assert [text for text, _, tail in _distribute_tail(pieces)] == [
        "denial of service attacks",
        "container escapes of note",
    ]
    # no preposition in the final member: nothing to distribute
    plain = [("terrorism", 0), ("genocide", 10), ("murder", 20)]
    assert all(tail is None for _, _, tail in _distribute_tail(plain))


def test_prepositional_phrases_are_expressible_as_composition_parts():
    """PPs are framing material, not just extraction debris."""

    composed = compose_expression([("head", "time"), ("pp", "of voting")])
    assert expression_surface(composed) == "time of voting"
    assert [part["role"] for part in composed["parts"]] == ["head", "pp"]
    contextual = compose_expression(
        [("relation", "study"), ("head", "topology"), ("context", "in a classroom")]
    )
    assert expression_surface(contextual) == "study topology in a classroom"


# --- arXiv taxonomy -------------------------------------------------------


TAXONOMY = """<html><body>
<h2 class="accordion-head" id="accordion-head-grp_cs">
  <button type="button">Computer Science</button></h2>
<div><h4>cs.AI <span>(Artificial Intelligence)</span></h4><p>Covers AI.</p>
<h4>cs.CR <span>(Cryptography and Security)</span></h4><p>Covers crypto.</p>
<h4>cs.CE <span>(Computational Engineering, Finance, and Science)</span></h4></div>
<h2 class="accordion-head" id="accordion-head-grp_physics">
  <button type="button">Physics</button></h2>
<div><h4>astro-ph.CO <span>(Cosmology and Nongalactic Astrophysics)</span></h4>
<h4>quant-ph <span>(Quantum Physics)</span></h4></div>
</body></html>
"""


@pytest.fixture()
def taxonomy(tmp_path):
    path = tmp_path / "taxonomy.html"
    path.write_text(TAXONOMY, encoding="utf-8")
    return ArxivTaxonomySource(path, source_id="taxonomy.test")


def test_taxonomy_parses_codes_names_and_hierarchy(taxonomy):
    categories = {category.code: category for category in taxonomy.categories()}
    assert categories["cs.AI"].name == "Artificial Intelligence"
    assert categories["cs.AI"].group == "computer_science"
    assert categories["cs.AI"].archive == "cs"
    # group is positional, not derivable from the code
    assert categories["astro-ph.CO"].group == "physics"
    assert categories["astro-ph.CO"].archive == "astro-ph"


def test_taxonomy_categories_without_a_subcode_still_parse(taxonomy):
    codes = {category.code for category in taxonomy.categories()}
    assert "quant-ph" in codes, "top-level archives have no dotted subcode"


def test_taxonomy_emits_names_not_codes(taxonomy):
    result = taxonomy.extract()
    surfaces = {candidate.surface for candidate in result.candidates}
    assert "artificial intelligence" in surfaces
    assert "cs.AI" not in surfaces
    assert all(candidate.projection_source == "ontological" for candidate in result.candidates)


def test_taxonomy_carries_two_level_grouping(taxonomy):
    candidate = next(c for c in taxonomy.extract().candidates if c.surface == "quantum physics")
    assert candidate.groups == {"arxiv_group": "physics", "arxiv_archive": "quant_ph"}
    assert candidate.source_ref == "quant-ph"


def test_taxonomy_rejects_overlong_names_with_a_reason(taxonomy):
    result = taxonomy.extract()
    assert result.reason_counts().get("too-many-words") == 1
    assert any("Computational Engineering" in j.surface for j in result.rejected)


def test_taxonomy_layout_change_fails_loudly(tmp_path):
    empty = tmp_path / "moved.html"
    empty.write_text("<html><body><p>no categories here</p></body></html>", encoding="utf-8")
    with pytest.raises(StimulusSourceError, match="layout"):
        ArxivTaxonomySource(empty, source_id="taxonomy.moved").categories()
