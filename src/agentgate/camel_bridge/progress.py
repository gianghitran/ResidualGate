
from __future__ import annotations

import sys
import time

__all__ = ["Phase"]


class Phase:

    def __init__(self, label: str, total: int | None = None, *, every: int = 10,
                 enabled: bool = True, stream=None) -> None:
        self.label, self.total, self.every = label, total, every
        self.stream = stream or sys.stderr
        self.enabled = enabled and not self.stream.closed
        self.tty = self.enabled and self.stream.isatty()
        self.n = 0
        self.counters: dict[str, int] = {}
        self.started = time.monotonic()
        if self.enabled:
            self._write(f"[{self.label}] start"
                        + (f" · {self.total} items" if self.total else "") + "\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _line(self) -> str:
        of = f"/{self.total}" if self.total else ""
        counts = " · ".join(f"{k} {v}" for k, v in self.counters.items())
        elapsed = time.monotonic() - self.started
        return (f"[{self.label}] {self.n}{of}"
                + (f" · {counts}" if counts else "")
                + f" · {elapsed:.0f}s")

    def step(self, **counters: int) -> None:
        self.n += 1
        for key, value in counters.items():
            self.counters[key] = self.counters.get(key, 0) + value
        if not self.enabled:
            return
        if self.tty:
            self._write("\r\033[K" + self._line())
        elif self.n % self.every == 0:
            self._write(self._line() + "\n")

    def done(self, note: str = "") -> None:
        if not self.enabled:
            return
        prefix = "\r\033[K" if self.tty else ""
        self._write(f"{prefix}{self._line()} · done{(' · ' + note) if note else ''}\n")
