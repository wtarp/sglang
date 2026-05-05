from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PinInfo:
    expires_at: float
    min_protected_len: int


class ContinuumPinManager:
    """Tracks temporary request-level pin state for Continuum scheduling."""

    def __init__(self) -> None:
        self._pinned: dict[str, PinInfo] = {}

    def is_pinned(self, rid: str) -> bool:
        info = self._pinned.get(rid)
        if info is None:
            return False
        if time.time() >= info.expires_at:
            self._pinned.pop(rid, None)
            return False
        return True

    def get_pin_info(self, rid: str) -> PinInfo | None:
        if not self.is_pinned(rid):
            return None
        return self._pinned.get(rid)

    def pin(self, rid: str, seconds: float, min_protected_len: int) -> bool:
        """Pin a request for `seconds`.

        Returns True if this call changed the pin state.
        """
        now = time.time()
        expires_at = now + max(seconds, 0.0)
        prev = self._pinned.get(rid)
        if prev is None or expires_at > prev.expires_at or min_protected_len > prev.min_protected_len:
            self._pinned[rid] = PinInfo(
                expires_at=expires_at,
                min_protected_len=max(min_protected_len, prev.min_protected_len if prev else 0),
            )
            return True
        return False

    def unpin_expired(self) -> list[str]:
        now = time.time()
        expired = [rid for rid, info in self._pinned.items() if now >= info.expires_at]
        for rid in expired:
            self._pinned.pop(rid, None)
        return expired

