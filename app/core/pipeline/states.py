"""Explicit pipeline states. One file's failure never stops the scan."""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    DISCOVERED = "discovered"
    IDENTIFIED = "identified"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    PARSING = "parsing"
    VALIDATING = "validating"
    CONVERTING = "converting"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"


ORDER = [State.DISCOVERED, State.IDENTIFIED, State.QUEUED, State.EXTRACTING, State.PARSING,
         State.VALIDATING, State.CONVERTING, State.EXPORTING, State.COMPLETED]

TERMINAL = {State.COMPLETED, State.FAILED, State.SKIPPED, State.UNSUPPORTED}


def is_terminal(state: State) -> bool:
    return state in TERMINAL
