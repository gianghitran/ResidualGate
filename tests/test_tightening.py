
from hypothesis import given, strategies as st

from agentgate.compose import Label, release_bounded, rescue, tighten
from agentgate.labels import DISPOSITIONS, PolicyLabel

labels = st.sampled_from(list(Label))
policy_labels = st.sampled_from(list(PolicyLabel))
dispositions = st.sampled_from(DISPOSITIONS)




@given(policy_labels)
def test_refusal_is_a_tightening_of_deferral(policy):
    assert policy.to_label("refuse") >= policy.to_label("defer")


@given(policy_labels, dispositions)
def test_no_disposition_ever_allows_an_undecided_action(policy, disposition):
    if policy is not PolicyLabel.ALLOW:
        assert policy.to_label(disposition) != Label.ALLOW


@given(policy_labels)
def test_dispositions_agree_wherever_the_policy_decides(policy):
    if policy is not PolicyLabel.UNDECIDED:
        assert policy.to_label("defer") == policy.to_label("refuse")




@given(labels, labels)
def test_never_loosens(policy, slm):
    assert tighten(policy, slm) >= policy


@given(labels, labels)
def test_soundness_preserved(policy, slm):
    if policy != Label.ALLOW:
        assert tighten(policy, slm) != Label.ALLOW


@given(labels, labels)
def test_is_max_and_idempotent(policy, slm):
    out = tighten(policy, slm)
    assert out == max(policy, slm)
    assert tighten(out, out) == out


@given(policy_labels, labels)
def test_tighten_is_inert_against_a_refusal_base_on_the_residual(policy, slm):
    if policy is PolicyLabel.UNDECIDED:
        assert tighten(policy.to_label("refuse"), slm) == Label.BLOCK




@given(policy_labels, labels)
def test_rescue_never_falls_below_the_deferral_floor(policy, slm):
    floor = policy.to_label("defer")
    out = rescue(floor, policy.to_label("refuse"), slm)
    assert out >= floor


@given(policy_labels, labels)
def test_rescue_never_allows(policy, slm):
    floor = policy.to_label("defer")
    if policy is not PolicyLabel.ALLOW:
        assert rescue(floor, policy.to_label("refuse"), slm) != Label.ALLOW


@given(policy_labels, labels)
def test_rescue_is_bounded_by_the_refusal_base(policy, slm):
    floor, refused = policy.to_label("defer"), policy.to_label("refuse")
    assert rescue(floor, refused, slm) <= refused


@given(policy_labels)
def test_rescue_only_moves_residual_items(policy):
    if policy is not PolicyLabel.UNDECIDED:
        floor, refused = policy.to_label("defer"), policy.to_label("refuse")
        for slm in Label:
            assert rescue(floor, refused, slm) == refused




@given(labels, labels)
def test_release_only_loosens_the_declared_edge(policy, slm):
    out = release_bounded(policy, slm, inside_envelope=True)
    if out < policy:
        assert policy == Label.REVIEW
        assert slm == Label.ALLOW
        assert out == Label.ALLOW


@given(labels, labels)
def test_release_outside_envelope_equals_strict(policy, slm):
    assert release_bounded(policy, slm, inside_envelope=False) == tighten(policy, slm)


def test_composition_happens_only_in_compose_py():
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentgate"
    offenders = []
    for path in src.rglob("*.py"):
        if path.name in ("compose.py", "labels.py"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bmax\s*\(\s*(policy|policy_label|defer_floor)\b", line):
                offenders.append(f"{path}:{lineno}")
    assert not offenders, f"composition outside compose.py: {offenders}"
