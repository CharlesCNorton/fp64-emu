"""Fake-tensor registrations so the ops trace under torch.compile."""
import torch

from ._ops import ops


def _noop(*args, **kwargs):
    return None


for _name in ("fp64_emu_mm", "fp64_emu_bmm", "fp64_emu_mm_nt", "fp64_emu_bmm_nt",
              "fp64_emu_mm_timed"):
    _packet = getattr(ops, _name, None)
    if _packet is None:
        continue
    _qual = getattr(_packet, "_qualified_op_name", None)
    if _qual is None:
        continue
    try:
        torch.library.register_fake(_qual)(_noop)
    except (AttributeError, RuntimeError):
        pass
