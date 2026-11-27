"""Shared-server deployment switch for the MM Nodepack file access policy.

``shared_server()`` reports whether this ComfyUI instance is used by more than
one principal (reachable from the network). In that case the file access
policy in :mod:`py.common` applies to queued (node) operations as well as to
every non-local route request: reads only from the approved ComfyUI
input/output roots, writes only under the output root.

Resolution order (documented in README.md, "Shared server file policy"):

1. The environment variable ``MM_NODEPACK_SHARED_SERVER`` –
   ``1/true/yes/on`` (case-insensitive) forces the policy on,
   ``0/false/no/off`` forces it off. An explicit setting always wins.
2. Otherwise: True when ComfyUI was started with a non-loopback ``--listen``
   address. ``127.0.0.1``, ``::1`` and ``localhost`` (incl. 127.0.0.0/8) are
   loopback; the value may be a comma-separated list and any non-loopback
   entry counts.

Importable without ComfyUI: ``comfy.cli_args`` is read lazily inside
``shared_server()``; a missing or unreadable ComfyUI runtime simply answers
False (private loopback behaviour).
"""
import os

_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def _is_loopback(entry: str) -> bool:
    """True for loopback addresses: 127.0.0.0/8, ::1, localhost."""
    e = entry.strip().lower()
    if e in ("::1", "localhost"):
        return True
    return e.startswith("127.")


def shared_server() -> bool:
    """True when the ComfyUI server is shared (several principals / reachable
    from the network) and the file access policy must apply."""
    env = os.environ.get("MM_NODEPACK_SHARED_SERVER")
    if env is not None:
        v = env.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        # unrecognized value: fall back to the --listen setting
    try:
        from comfy.cli_args import args
    except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests): private
        return False
    listen = str(getattr(args, "listen", "") or "")
    entries = [e.strip() for e in listen.split(",") if e.strip()]
    if not entries:
        return False
    return any(not _is_loopback(e) for e in entries)
