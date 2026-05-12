from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class PinInfo:
    expires_at: float
    min_protected_len: int
    node: Any
    swa_uuid_for_lock: str | None = None


class ContinuumPinManager:
    """Tracks temporary cache-node pin state for Continuum scheduling.

    A "pin" is represented by holding an extra lock_ref on a radix-tree node
    (via BasePrefixCache.inc_lock_ref / dec_lock_ref). This ensures the cached
    prefix survives across subsequent requests even if request IDs (rid) change.
    """

    def __init__(self) -> None:
        # key: id(node)
        self._pinned: dict[int, PinInfo] = {}

    def is_pinned_node(self, node: Any) -> bool:
        info = self._pinned.get(id(node))
        if info is None:
            return False
        if time.time() >= info.expires_at:
            self._pinned.pop(id(node), None)
            return False
        return True

    def get_pin_info_by_node(self, node: Any) -> PinInfo | None:
        node_key = id(node)
        info = self._pinned.get(node_key)
        if info is None:
            return None
        if time.time() >= info.expires_at:
            self._pinned.pop(node_key, None)
            return None
        return info

    def pin_node(
        self,
        node: Any,
        seconds: float,
        min_protected_len: int,
        *,
        swa_uuid_for_lock: str | None = None,
    ) -> bool:
        """Pin a cache node for `seconds`.

        Returns True if this call changed the pin state.
        """
        now = time.time()
        expires_at = now + max(seconds, 0.0)
        node_key = id(node)
        prev = self._pinned.get(node_key)
        if (
            prev is None
            or expires_at > prev.expires_at
            or min_protected_len > prev.min_protected_len
        ):
            self._pinned[node_key] = PinInfo(
                expires_at=expires_at,
                min_protected_len=max(
                    min_protected_len, prev.min_protected_len if prev else 0
                ),
                node=node,
                swa_uuid_for_lock=swa_uuid_for_lock
                or (prev.swa_uuid_for_lock if prev else None),
            )
            return True
        return False

    def pop_expired(self) -> list[PinInfo]:
        now = time.time()
        expired_keys = [k for k, info in self._pinned.items() if now >= info.expires_at]
        expired_infos: list[PinInfo] = []
        for k in expired_keys:
            info = self._pinned.pop(k, None)
            if info is not None:
                expired_infos.append(info)
        return expired_infos
