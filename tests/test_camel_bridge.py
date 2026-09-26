
from __future__ import annotations

import dataclasses

import pytest

from agentgate.camel_bridge.records import DenialSource, PolicyCallRecord, classify
from agentgate.labels import Label


@dataclasses.dataclass(frozen=True)
class Allowed:
    pass


@dataclasses.dataclass(frozen=True)
class Denied:
    reason: str


DATAFLOW = "send_money is state-changing and depends on private values ['x']."
FALLBACK = "No security policy matched for tool. Defaulting to denial."
TOOL_RULE = "The recipient does not come directly from the user."


class FakeEngine:

    def __init__(self, verdicts):
        self.policies = [("send_money", None)]
        self.no_side_effect_tools = {"get_balance"}
        self._verdicts = verdicts
        self.calls = []

    def check_policy(self, tool_name, kwargs, dependencies):
        self.calls.append(tool_name)
        return self._verdicts[tool_name]


CASES = {
    "get_balance": Allowed(),
    "sum": Denied(FALLBACK),
    "send_money": Denied(TOOL_RULE),
    "schedule_transaction": Denied(DATAFLOW),
    "read_file": Allowed(),
}


def _recorder(**kw):
    from agentgate.camel_bridge.engines import RecordingPolicyEngine

    return RecordingPolicyEngine(FakeEngine(CASES), trace_id="t", suite="banking",
                                 user_task_id="user_task_0", **kw)



def test_recording_returns_the_delegates_own_result_object():
    rec = _recorder()
    for tool in CASES:
        assert rec.check_policy(tool, {}, []) is CASES[tool]


def test_recording_does_not_reorder_or_drop_calls():
    rec = _recorder()
    for tool in CASES:
        rec.check_policy(tool, {}, [])
    assert rec.delegate.calls == list(CASES)
    assert [r.tool_name for r in rec.records] == list(CASES)


def test_recording_preserves_the_engines_public_surface():
    rec = _recorder()
    assert rec.policies == rec.delegate.policies
    assert rec.no_side_effect_tools == rec.delegate.no_side_effect_tools



def test_builtin_denials_are_recorded_but_not_counted():
    rec = _recorder(tool_names=["get_balance", "send_money", "schedule_transaction", "read_file"])
    for tool in CASES:
        rec.check_policy(tool, {}, [])

    builtin = next(r for r in rec.records if r.tool_name == "sum")
    assert builtin.denied and builtin.denial_source is DenialSource.COVERAGE_FALLBACK
    assert not builtin.is_tool
    assert not builtin.is_enforced_residual, "a discarded denial must not count as residual"

    assert {r.tool_name for r in rec.residual_records()} == {"send_money", "schedule_transaction"}


def test_without_an_inventory_every_call_counts():
    rec = _recorder()
    rec.check_policy("sum", {}, [])
    assert rec.records[0].is_tool



@pytest.mark.parametrize("reason,expected", [
    (DATAFLOW, DenialSource.DATAFLOW_GUARD),
    (FALLBACK, DenialSource.COVERAGE_FALLBACK),
    (TOOL_RULE, DenialSource.TOOL_POLICY),
])
def test_denials_are_attributed_to_the_right_branch(reason, expected):
    got = classify(denied=True, reason=reason, tool_name="send_money", no_side_effect_tools=frozenset())
    assert got is expected and got.is_residual


def test_allow_list_hits_are_not_evidence_about_the_policies():
    nse = frozenset({"get_balance"})
    assert classify(denied=False, reason=None, tool_name="get_balance",
                    no_side_effect_tools=nse) is DenialSource.NO_SIDE_EFFECT
    assert classify(denied=False, reason=None, tool_name="send_money",
                    no_side_effect_tools=nse) is DenialSource.ALLOWED


def test_every_source_lands_on_exactly_one_partition():
    for source in DenialSource:
        assert source.partition in {"policy_decidable", "policy_residual", "policy_error"}
        assert source.is_residual == (source.partition == "policy_residual")


def test_a_policy_that_raises_is_neither_decidable_nor_residual():
    got = classify(denied=True, reason="KeyError: 'recipient'", tool_name="send_money",
                   no_side_effect_tools=frozenset(), raised=True)
    assert got is DenialSource.POLICY_ERROR
    assert not got.is_residual and got.partition == "policy_error"


def test_a_raising_policy_is_recorded_and_then_re_raised():
    from agentgate.camel_bridge.engines import RecordingPolicyEngine

    class Exploding(FakeEngine):
        def check_policy(self, tool_name, kwargs, dependencies):
            raise KeyError("recipient")

    rec = RecordingPolicyEngine(Exploding(CASES), trace_id="t", suite="banking",
                                user_task_id="u", tool_names=("send_money",))
    with pytest.raises(KeyError):
        rec.check_policy("send_money", {}, [])
    assert len(rec.records) == 1
    assert rec.records[0].raised and rec.records[0].denial_source is DenialSource.POLICY_ERROR
    assert not rec.records[0].is_enforced_residual


def test_record_serialises_partition_and_source_as_strings():
    r = PolicyCallRecord(trace_id="t", suite="banking", user_task_id="u", injection_task_id=None,
                         attack_name=None, call_index=0, tool_name="send_money", denied=True,
                         denial_source=DenialSource.TOOL_POLICY, reason=TOOL_RULE,
                         gateway_label=Label.REVIEW.name)
    d = r.to_json()
    assert d["denial_source"] == "tool_policy" and d["partition"] == "policy_residual"
    assert d["gateway_label"] == "REVIEW"



class FakeAllowed:
    pass


def _gateway(kind, review_execution="halt", tools=("send_money", "get_balance")):
    from agentgate.camel_bridge.engines import DeferEngine, RecordingPolicyEngine, RefuseEngine

    delegate = FakeEngine(CASES)
    rec = RecordingPolicyEngine(delegate, trace_id="t", suite="banking",
                                user_task_id="u", tool_names=tools)
    cls = DeferEngine if kind == "defer" else RefuseEngine
    return cls(FakeEngine(CASES), recorder=rec, review_execution=review_execution), rec


def test_gateway_records_carry_is_tool():
    gw, rec = _gateway("defer")
    for tool in ("send_money", "sum"):
        gw.check_policy(tool, {}, [])
    by_name = {r.tool_name: r for r in rec.records}
    assert by_name["send_money"].is_tool
    assert not by_name["sum"].is_tool


def test_a_discarded_builtin_denial_does_not_halt_the_run():
    gw, _ = _gateway("refuse")
    gw.check_policy("sum", {}, [])
    assert not gw.halted


def test_an_enforced_denial_halts_the_run():
    gw, _ = _gateway("refuse")
    gw.check_policy("send_money", {}, [])
    assert gw.halted


def test_the_two_dispositions_bill_the_same_residual_differently():
    defer, defer_rec = _gateway("defer", review_execution="proceed")
    refuse, _ = _gateway("refuse")
    for gw in (defer, refuse):
        gw.check_policy("send_money", {}, [])

    assert not defer.halted and refuse.halted
    assert [r.gateway_label for r in defer_rec.records] == ["REVIEW"]


def test_halting_defer_is_the_pessimistic_bound():
    defer, _ = _gateway("defer", review_execution="halt")
    defer.check_policy("send_money", {}, [])
    assert defer.halted


def test_rsr_is_a_property_of_the_policy_not_the_disposition():
    from agentgate.camel_bridge.metrics import call_metrics

    rates = []
    for kind in ("defer", "refuse"):
        gw, rec = _gateway(kind, review_execution="proceed")
        for tool in CASES:
            gw.check_policy(tool, {}, [])
        rates.append(call_metrics(rec.records).rsr)
    assert rates[0] == rates[1]



def test_breach_rate_ignores_traces_where_no_attack_ran():
    from agentgate.camel_bridge.metrics import TraceOutcome, trace_metrics

    m = trace_metrics([
        TraceOutcome("a", "banking", True, True, breached=None),
        TraceOutcome("b", "banking", True, True, breached=False),
        TraceOutcome("c", "banking", True, False, breached=True),
    ])
    assert m.breach_rate == 0.5, "denominator must be the two attacked traces"
    assert m.n == 3


def test_a_trace_that_fails_without_the_policy_is_excluded_entirely():
    from agentgate.camel_bridge.metrics import TraceOutcome, trace_metrics

    m = trace_metrics([
        TraceOutcome("a", "banking", True, False, breached=True),
        TraceOutcome("b", "banking", False, False, breached=True),
    ])
    assert m.n == 1 and m.n_excluded == 1
    assert m.ulr == 1.0 and m.breach_rate == 1.0


def test_finishing_is_not_succeeding():
    from agentgate.camel_bridge.metrics import TraceOutcome, trace_metrics

    m = trace_metrics([TraceOutcome("a", "banking", True, False, halted_by_gateway=False)])
    assert m.utility == 0.0 and m.ulr == 1.0



def _composed(mode, model, review_execution="proceed", envelope=None, consult="residual"):
    from agentgate.camel_bridge.engines import ComposedEngine, RecordingPolicyEngine

    rec = RecordingPolicyEngine(FakeEngine(CASES), trace_id="t", suite="banking",
                                user_task_id="u", tool_names=("send_money", "get_balance",
                                                              "schedule_transaction", "read_file"))
    return ComposedEngine(FakeEngine(CASES), model=model, mode=mode, recorder=rec,
                          review_execution=review_execution, envelope=envelope,
                          consult=consult), rec


def _labels(engine, rec):
    for tool in CASES:
        engine.check_policy(tool, {}, [])
    return [r.gateway_label for r in rec.records]


ALWAYS_ALLOW = lambda t, k, d=(), r=None: Label.ALLOW
ALWAYS_BLOCK = lambda t, k, d=(), r=None: Label.BLOCK


def test_a_model_that_always_allows_leaves_the_gateway_exactly_at_the_floor():
    from agentgate.camel_bridge.engines import DeferEngine, RecordingPolicyEngine

    floor_rec = RecordingPolicyEngine(FakeEngine(CASES), trace_id="t", suite="banking",
                                      user_task_id="u", tool_names=("send_money", "get_balance",
                                                                    "schedule_transaction", "read_file"))
    floor = DeferEngine(FakeEngine(CASES), recorder=floor_rec, review_execution="proceed")
    strict, strict_rec = _composed("strict", ALWAYS_ALLOW)

    assert _labels(strict, strict_rec) == _labels(floor, floor_rec)


def test_a_model_that_always_blocks_stops_the_run_and_never_unsounds_it():
    strict, rec = _composed("strict", ALWAYS_BLOCK)
    strict.check_policy("send_money", {}, [])
    assert rec.records[0].gateway_label == "BLOCK" and strict.halted



def test_by_default_the_model_is_not_consulted_where_the_policy_already_decided():
    strict, rec = _composed("strict", ALWAYS_BLOCK)
    strict.check_policy("get_balance", {}, [])
    assert rec.records[0].gateway_label == "ALLOW"
    assert rec.records[0].model_label is None, "the model was not asked"
    assert not strict.halted


def test_consult_all_lets_the_model_override_a_decided_call():
    strict, rec = _composed("strict", ALWAYS_BLOCK, consult="all")
    strict.check_policy("get_balance", {}, [])
    assert rec.records[0].gateway_label == "BLOCK" and rec.records[0].model_label == "BLOCK"


def test_an_unknown_consult_mode_is_rejected():
    with pytest.raises(ValueError, match="consult"):
        _composed("strict", ALWAYS_ALLOW, consult="sometimes")


def test_rescue_lifts_a_refusal_only_as_far_as_review():
    rescue, rec = _composed("rescue", ALWAYS_ALLOW)
    rescue.check_policy("send_money", {}, [])
    assert rec.records[0].gateway_label == "REVIEW", "rescue must never reach ALLOW"


def test_rescue_leaves_a_refusal_alone_when_the_model_is_not_confident():
    rescue, rec = _composed("rescue", lambda t, k, d=(), r=None: Label.REVIEW)
    rescue.check_policy("send_money", {}, [])
    assert rec.records[0].gateway_label == "BLOCK"


def test_crossing_the_soundness_floor_must_be_declared():
    with pytest.raises(ValueError, match="envelope"):
        _composed("bounded", ALWAYS_ALLOW)


def test_bounded_loosens_only_inside_the_declared_envelope():
    from agentgate.camel_bridge.engines import envelope_from

    inside, rec_in = _composed("bounded", ALWAYS_ALLOW, envelope=envelope_from(["send_money"]))
    inside.check_policy("send_money", {}, [])
    assert rec_in.records[0].gateway_label == "ALLOW"

    outside, rec_out = _composed("bounded", ALWAYS_ALLOW, envelope=envelope_from(["read_file"]))
    outside.check_policy("send_money", {}, [])
    assert rec_out.records[0].gateway_label == "REVIEW"


def test_the_model_label_is_recorded_next_to_the_gateway_label():
    strict, rec = _composed("strict", ALWAYS_ALLOW)
    strict.check_policy("send_money", {}, [])
    assert rec.records[0].model_label == "ALLOW"
    assert rec.records[0].gateway_label == "REVIEW"



def test_a_backend_that_raises_becomes_review_and_is_counted():
    from agentgate.backends.base import BackendConfig
    from agentgate.camel_bridge.slm import SLMGate

    class Exploding:
        name = "boom"

        def generate(self, prompts, config):
            raise RuntimeError("cuda is on fire")

    gate = SLMGate(Exploding(), BackendConfig(model="x"))
    assert gate("send_money", {}) is Label.REVIEW
    assert gate.failures == 1 and gate.verdicts[0]["status"] == "error"


def test_structured_view_carries_every_decision_fact():
    from agentgate.camel_bridge.slm import ArgumentView, CallView

    view = CallView(
        tool="send_money",
        arguments=(ArgumentView("recipient", "US13", False, True, ()),
                   ArgumentView("amount", "100.0", True, True, ())),
        non_public_dependencies=("balance: 1810.0",), goal="Pay the rent.")

    text = view.structured()
    assert "send_money" in text and "US13" in text and "100.0" in text
    assert "balance: 1810.0" in text and "Pay the rent." in text
    assert "{" in text


def test_the_model_is_not_asked_about_builtins():
    from agentgate.camel_bridge.engines import ComposedEngine, RecordingPolicyEngine

    asked = []

    def spy(tool_name, kwargs, dependencies=(), reason=None):
        asked.append(tool_name)
        return Label.BLOCK

    rec = RecordingPolicyEngine(FakeEngine(CASES), trace_id="t", suite="banking",
                                user_task_id="u", tool_names=("send_money",))
    gw = ComposedEngine(FakeEngine(CASES), model=spy, mode="strict", recorder=rec,
                        review_execution="proceed", tool_names=("send_money",),
                        consult="all")
    gw.check_policy("sum", {}, [])
    gw.check_policy("send_money", {}, [])
    assert asked == ["send_money"]


def test_public_and_nobody_are_not_the_same_readers():
    import sys

    from agentgate.camel_bridge.slm import ANYONE, CallView

    class Public:
        pass

    import camel.capabilities as caps

    original = {name: getattr(caps, name)
                for name in ("get_all_readers", "is_trusted", "is_public")}
    try:
        caps.is_trusted = lambda value: False
        caps.is_public = lambda value: True

        caps.get_all_readers = lambda value: (Public(), None)
        assert CallView.from_camel("send_money", {"x": object()}).arguments[0].readers == (ANYONE,)

        caps.get_all_readers = lambda value: (frozenset({"bob"}), None)
        arg = CallView.from_camel("send_money", {"x": object()}).arguments[0]
        assert arg.readers == ("bob",) and arg.restricted_to == ("bob",)
    finally:
        for name, fn in original.items():
            setattr(caps, name, fn)


def test_an_unrestricted_value_is_not_marked_restricted():
    from agentgate.camel_bridge.slm import ANYONE, ArgumentView

    arg = ArgumentView("r", "US13", False, True, (ANYONE,))
    assert arg.restricted_to == ()


def test_the_policys_own_reason_reaches_the_model():
    seen = {}

    def spy(tool_name, kwargs, dependencies=(), reason=None):
        seen["reason"] = reason
        return Label.ALLOW

    engine, _ = _composed("strict", spy)
    engine.check_policy("send_money", {}, [])
    assert seen["reason"] == TOOL_RULE


def test_the_view_shows_why_the_policy_refused():
    from agentgate.camel_bridge.slm import ArgumentView, CallView

    view = CallView("send_money", (ArgumentView("r", "US13", False, True, ()),),
                    goal="Pay the rent.", policy_reason=TOOL_RULE)
    assert TOOL_RULE in view.structured()


def test_breach_is_not_gated_on_the_task_being_doable():
    from agentgate.camel_bridge.metrics import TraceOutcome, trace_metrics

    m = trace_metrics([
        TraceOutcome("a", "banking", succeeded_without_policy=False,
                     succeeded=False, breached=True),
        TraceOutcome("b", "banking", succeeded_without_policy=False,
                     succeeded=False, breached=False),
    ])
    assert m.n == 0 and m.n_excluded == 2, "both are excluded from utility"
    assert m.utility is None and m.ulr is None
    assert m.n_attacked == 2 and m.breach_rate == 0.5, "but both still testify about security"


def test_bounded_refuses_an_empty_envelope_through_the_runner():
    from agentgate.camel_bridge.replay import SYSTEMS_BY_NAME, _build_engine

    with pytest.raises(ValueError, match="envelope"):
        _build_engine(SYSTEMS_BY_NAME["b3b"], FakeEngine(CASES), recorder=None,
                      envelope_tools=(), model=ALWAYS_ALLOW)


def test_a_gateway_records_a_policy_that_raises_and_re_raises_it():
    import pytest

    from agentgate.camel_bridge.engines import RefuseEngine, RecordingPolicyEngine
    from agentgate.camel_bridge.records import DenialSource

    class Raising(FakeEngine):
        def check_policy(self, tool_name, kwargs, dependencies):
            raise KeyError("recipient")

    rec = RecordingPolicyEngine(FakeEngine(CASES), trace_id="t", suite="banking",
                                user_task_id="u", tool_names=("send_money",))
    gw = RefuseEngine(Raising(CASES), recorder=rec)

    with pytest.raises(KeyError):
        gw.check_policy("send_money", {}, [])

    assert len(rec.records) == 1, "the raise must leave a record behind"
    record = rec.records[0]
    assert record.raised and record.denial_source is DenialSource.POLICY_ERROR
    assert record.partition == "policy_error", "neither decidable nor residual"
    assert "recipient" in record.reason, "the retry reads the argument name off this"
    assert gw.halted, "a raising policy ends the run as surely as a denial"
