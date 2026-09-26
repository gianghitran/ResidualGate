
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, Sequence

from ..compose import release_bounded, rescue, tighten
from ..labels import Disposition, Label, PolicyLabel
from .records import DenialSource, PolicyCallRecord, classify

if TYPE_CHECKING:
    from camel.interpreter.value import CaMeLValue

__all__ = [
    "RecordingPolicyEngine",
    "GatewayEngine",
    "DeferEngine",
    "RefuseEngine",
    "ComposedEngine",
    "ReviewExecution",
]

ReviewExecution = Literal["halt", "proceed", "oracle"]


@dataclasses.dataclass(frozen=True)
class _RaisedVerdict:

    reason: str


def _label_of(result: Any) -> PolicyLabel:
    return PolicyLabel.ALLOW if type(result).__name__ == "Allowed" else PolicyLabel.UNDECIDED


def _reason_of(result: Any) -> str | None:
    return getattr(result, "reason", None)


def payload_readers(kwargs: "Mapping[str, CaMeLValue] | None") -> str | None:
    if not kwargs:
        return None
    try:
        from camel.capabilities import readers as _readers
        from camel.capabilities.utils import get_all_readers
    except Exception:
        return None
    seen = False
    for name, value in kwargs.items():
        try:
            value_readers, _ = get_all_readers(value)
        except Exception:
            continue
        seen = True
        if not isinstance(value_readers, _readers.Public):
            return "restricted"
    return "public" if seen else None


class _Wrapper:

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.policies = delegate.policies
        self.no_side_effect_tools = delegate.no_side_effect_tools

    @property
    def delegate(self) -> Any:
        return self._delegate


class RecordingPolicyEngine(_Wrapper):

    def __init__(self, delegate: Any, *, trace_id: str, suite: str, user_task_id: str,
                 injection_task_id: str | None = None, attack_name: str | None = None,
                 tool_names: Iterable[str] = ()) -> None:
        super().__init__(delegate)
        self._meta = dict(trace_id=trace_id, suite=suite, user_task_id=user_task_id,
                          injection_task_id=injection_task_id, attack_name=attack_name)
        self._tool_names = frozenset(tool_names)
        self.records: list[PolicyCallRecord] = []

    def check_policy(self, tool_name: str, kwargs: "Mapping[str, CaMeLValue]",
                     dependencies: "Iterable[CaMeLValue]") -> Any:
        try:
            result = self._delegate.check_policy(tool_name, kwargs, dependencies)
        except Exception as exc:
            self.note(tool_name, _RaisedVerdict(f"{type(exc).__name__}: {exc}"),
                      raised=True, kwargs=kwargs)
            raise
        self.note(tool_name, result, kwargs=kwargs)
        return result

    def note(self, tool_name: str, result: Any, *, gateway_label: Label | None = None,
             model_label: Label | None = None, raised: bool = False,
             kwargs: "Mapping[str, CaMeLValue] | None" = None) -> PolicyCallRecord:
        denied = raised or _label_of(result) is not PolicyLabel.ALLOW
        record = PolicyCallRecord(
            call_index=len(self.records), tool_name=tool_name, denied=denied,
            denial_source=classify(denied=denied, reason=_reason_of(result), tool_name=tool_name,
                                   no_side_effect_tools=frozenset(self.no_side_effect_tools),
                                   raised=raised),
            reason=_reason_of(result), is_tool=self._is_tool(tool_name), raised=raised,
            gateway_label=None if gateway_label is None else gateway_label.name,
            model_label=None if model_label is None else model_label.name,
            payload_readers=payload_readers(kwargs), **self._meta)
        self.records.append(record)
        return record

    def _is_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_names if self._tool_names else True

    def residual_records(self) -> list[PolicyCallRecord]:
        return [r for r in self.records if r.is_enforced_residual]


class GatewayEngine(_Wrapper):

    def __init__(self, delegate: Any, *, disposition: Disposition = "defer",
                 review_execution: ReviewExecution = "halt",
                 reviewer: Callable[[str, Mapping[str, Any]], bool] | None = None,
                 recorder: RecordingPolicyEngine | None = None) -> None:
        super().__init__(delegate)
        if review_execution == "oracle" and reviewer is None:
            raise ValueError("review_execution='oracle' needs a reviewer callable")
        self._disposition = disposition
        self._review_execution = review_execution
        self._reviewer = reviewer
        self._recorder = recorder
        self.decisions: list[Label] = []
        self.halted = False

    def decide(self, policy: PolicyLabel, tool_name: str, kwargs: "Mapping[str, CaMeLValue]",
               dependencies: "Iterable[CaMeLValue]" = (),
               reason: str | None = None) -> tuple[Label, Label | None]:
        return policy.to_label(self._disposition), None

    def check_policy(self, tool_name: str, kwargs: "Mapping[str, CaMeLValue]",
                     dependencies: "Iterable[CaMeLValue]") -> Any:
        from camel.security_policy import Allowed, Denied

        try:
            result = self._delegate.check_policy(tool_name, kwargs, dependencies)
        except Exception as exc:
            if self._recorder is not None:
                self._recorder.note(tool_name, _RaisedVerdict(f"{type(exc).__name__}: {exc}"),
                                    raised=True, kwargs=kwargs)
            self.halted = True
            raise
        policy = _label_of(result)
        label, model_label = self.decide(policy, tool_name, kwargs, dependencies,
                                         _reason_of(result))
        self.decisions.append(label)

        record = None
        if self._recorder is not None:
            record = self._recorder.note(tool_name, result, gateway_label=label,
                                         model_label=model_label, kwargs=kwargs)

        if label is Label.ALLOW:
            return Allowed()
        verdict = (Denied(_reason_of(result) or f"{tool_name} blocked by gateway")
                   if label is Label.BLOCK
                   else self._dispose_review(tool_name, kwargs, result, Allowed, Denied))

        if isinstance(verdict, Denied) and (record is None or record.is_tool):
            self.halted = True
        return verdict

    def _dispose_review(self, tool_name: str, kwargs: "Mapping[str, CaMeLValue]",
                        result: Any, Allowed: type, Denied: type) -> Any:
        if self._review_execution == "proceed":
            return Allowed()
        if self._review_execution == "halt":
            return Denied(f"{tool_name} queued for human review (not executed under review_execution='halt')")
        approved = self._reviewer(tool_name, kwargs)
        return Allowed() if approved else Denied(f"{tool_name} rejected by oracle reviewer")


class DeferEngine(GatewayEngine):

    def __init__(self, delegate: Any, **kw: Any) -> None:
        kw.pop("disposition", None)
        super().__init__(delegate, disposition="defer", **kw)


class RefuseEngine(GatewayEngine):

    def __init__(self, delegate: Any, **kw: Any) -> None:
        kw.pop("disposition", None)
        super().__init__(delegate, disposition="refuse", **kw)


ComposeMode = Literal["strict", "bounded", "rescue", "model_only"]


@dataclasses.dataclass(frozen=True)
class _EnvelopeCheck:

    inside_tools: frozenset[str]

    def __call__(self, tool_name: str) -> bool:
        return tool_name in self.inside_tools


class ComposedEngine(GatewayEngine):

    def __init__(self, delegate: Any, *, model: Callable[..., Label],
                 mode: ComposeMode = "strict", envelope: _EnvelopeCheck | None = None,
                 tool_names: Iterable[str] = (), consult: str = "residual", **kw: Any) -> None:
        if mode == "bounded" and envelope is None:
            raise ValueError("mode='bounded' crosses the soundness floor and needs an explicit envelope (§7.4)")
        disposition: Disposition = "refuse" if mode == "rescue" else kw.pop("disposition", "defer")
        kw.pop("disposition", None)
        super().__init__(delegate, disposition=disposition, **kw)
        self._model = model
        self._mode = mode
        self._envelope = envelope
        self._tool_names = frozenset(tool_names)
        if consult not in ("residual", "all"):
            raise ValueError(f"unknown consult mode {consult!r}")
        self._consult = consult

    def decide(self, policy: PolicyLabel, tool_name: str, kwargs: "Mapping[str, CaMeLValue]",
               dependencies: "Iterable[CaMeLValue]" = (),
               reason: str | None = None) -> tuple[Label, Label | None]:
        if self._tool_names and tool_name not in self._tool_names:
            return policy.to_label(self._disposition), None
        if self._consult == "residual" and policy is PolicyLabel.ALLOW:
            return policy.to_label(self._disposition), None
        model_label = self._model(tool_name, kwargs, dependencies, reason)
        if self._mode == "strict":
            return tighten(policy.to_label("defer"), model_label), model_label
        if self._mode == "bounded":
            inside = self._envelope(tool_name)
            return release_bounded(policy.to_label("defer"), model_label, inside), model_label
        if self._mode == "model_only":
            return model_label, model_label
        return rescue(policy.to_label("defer"), policy.to_label("refuse"), model_label), model_label


def envelope_from(tools: Sequence[str]) -> _EnvelopeCheck:
    return _EnvelopeCheck(frozenset(tools))
